from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

# ── Sentinel for hard loop stop ────────────────────────────────────────────────
# OCRToolExecutor sets this token in tool results when hard-stop is desired
# (e.g., after reaching figure limit).
HARD_STOP_TOKEN = "__HARD_STOP__"

import config

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("❌ openai Paket fehlt. Bitte installieren: pip install openai")


# ── Client creation ────────────────────────────────────────────────────────────

_CLIENT_INSTANCE: "OpenAI" | None = None

# Creates or returns an existing OpenAI client (singleton).
def create_openai_client() -> "OpenAI":
    """
    Creates or returns an existing OpenAI client (singleton).
    Prevents accumulation of connection pools in memory.
    """
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = OpenAI(
            base_url=config.LM_STUDIO_BASE_URL,
            api_key=config.LM_STUDIO_API_KEY,
            timeout=config.LM_STUDIO_TIMEOUT,
        )
    return _CLIENT_INSTANCE


# ── Server check ──────────────────────────────────────────────────────────────

# Quick ping check if LM Studio is reachable.
def is_server_reachable() -> bool:
    """Quick ping check if LM Studio is reachable."""
    try:
        url = f"{config.LM_STUDIO_BASE_URL}/models"
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# Searches for the lms CLI binary in known paths.
def find_lms_tool() -> str | None:
    """Searches for the lms CLI binary in known paths."""
    candidates = [
        shutil.which("lms"),
        str(Path.home() / ".lmstudio" / "bin" / "lms"),
        str(Path.home() / ".cache" / "lm-studio" / "bin" / "lms"),
        "/usr/local/bin/lms",
        "/opt/homebrew/bin/lms",
    ]
    for p in candidates:
        if p and Path(p).exists() and Path(p).stat().st_size > 0:
            return p
    return None


# Starts LM Studio server if not reachable. Waits up to 30 seconds.
def ensure_lm_studio_running() -> bool:
    """Starts LM Studio server if not reachable. Waits up to 30 seconds."""
    if is_server_reachable():
        return True
    lms = find_lms_tool()
    if not lms:
        print("❌ 'lms' CLI nicht gefunden. LM Studio manuell starten.")
        return False
    print(f"⚠️ LM Studio offline. Starte via {lms}...")
    try:
        subprocess.Popen([lms, "server", "start"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Start fehlgeschlagen: {e}")
        return False
    for _ in range(15):
        time.sleep(2)
        if is_server_reachable():
            print("✅ LM Studio Server bereit.")
            return True
    print("❌ LM Studio antwortet nach 30s nicht.")
    return False


# ── Model switching ───────────────────────────────────────────────────────────

# Checks if model is already loaded. If not: loads via lms CLI.
def switch_model(client: "OpenAI", search_term: str, load_string: str) -> str:
    """
    Checks if model is already loaded. If not: loads via lms CLI.
    Returns actual model ID.
    """
    try:
        models = client.models.list()
        for m in models.data:
            if search_term.lower() in m.id.lower():
                print(f"   ✅ Modell bereits aktiv: {m.id}")
                return m.id
    except Exception:
        pass

    lms = find_lms_tool()
    if lms:
        print(f"🔄 Lade Modell: {load_string}...")
        try:
            subprocess.run(
                [lms, "load", load_string],
                input="\n", text=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=300,
            )
        except Exception as e:
            print(f"   ⚠️ Lade-Fehler (ignoriert): {e}")
        time.sleep(5)
    else:
        print(f"   ⚠️ 'lms' nicht gefunden. Modell muss manuell geladen sein.")

    return load_string


# Unloads a model via lms CLI. Fire-and-forget.
def unload_model(model_id: str) -> None:
    """Unloads a model via lms CLI. Fire-and-forget."""
    lms = find_lms_tool()
    if lms:
        print(f"👋 Entlade Modell: {model_id}...")
        try:
            subprocess.run(
                [lms, "unload", model_id],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception:
            pass
        time.sleep(2)
    gc.collect()


# ── LLM calls ─────────────────────────────────────────────────────────────────

# Wrapper around chat.completions.create with exponential retry backoff.
def robust_chat_completion(
    client: "OpenAI",
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float = 0.1,
    retries: int = 3,
    **kwargs,
) -> str:
    """
    Wrapper around chat.completions.create with two independent retry layers.

    Layer 1 — empty output (inner loop, 1 automatic retry):
        If the model returns an empty string after stripping think-blocks
        (e.g. it exhausted all tokens inside a <think> block), the call is
        retried once after a short pause. This does NOT consume an
        API-error retry slot.

    Layer 2 — API / connection errors (outer loop, *retries* attempts):
        APIConnectionError / RateLimitError → exponential backoff (2^n s).
        APIError (e.g. model crash, code 400) → linear backoff (5n s).

    Returns '[ERROR: ...]' on final failure (never raises).
    """
    for attempt in range(retries):
        try:
            # ── Inner loop: one silent retry on empty output ──────────────────
            # Handles the case where the model burns its entire token budget
            # inside a <think> block and returns nothing useful.
            for _empty_try in range(2):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                raw = response.choices[0].message.content or ""
                text = strip_think_blocks(raw)

                if not text.strip() and _empty_try == 0:
                    finish = getattr(response.choices[0], "finish_reason", "unknown")
                    print(
                        f"     ⚠️ Leere Antwort vom Modell "
                        f"(finish_reason={finish}). Einmal neu versuchen in 3s..."
                    )
                    time.sleep(3)
                    continue  # inner retry — does not increment *attempt*

                break  # has content, or already retried once → accept result

            return text

        except (APIConnectionError, RateLimitError) as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"     ⚠️ Verbindungsfehler ({e}). Warte {wait}s...")
                time.sleep(wait)
            else:
                return f"[ERROR: {e}]"
        except APIError as e:
            # APIError (e.g. code 400, model crash) — also retry with backoff.
            # Model needs some time to recover after a crash.
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)  # 5s, 10s — longer waits than connection errors
                print(f"     ⚠️ API-Fehler (Code {getattr(e, 'status_code', '?')}): {e}. Retry {attempt + 2}/{retries} in {wait}s...")
                time.sleep(wait)
            else:
                return f"[ERROR: API {e}]"
        except Exception as e:
            return f"[ERROR: {e}]"
    return "[ERROR: Timeout]"


# Agent-based chat loop with tool use.
def agentic_chat_completion(
    client: "OpenAI",
    model: str,
    messages: list[dict],
    tools: list[dict],
    tool_executor: callable,
    max_iterations: int | None = None,
    temperature: float = 0.1,
    log_callback: callable = None,
    **kwargs,
) -> tuple[str, list[dict]]:
    """
    Agent-based chat loop with tool use.

    Returns (final_text, tool_call_log).
    tool_executor(name, args) -> str is called for each tool call.
    tool_call_log: [{'name': str, 'args': dict, 'result_preview': str}]
    """
    if not config.AGENT_TOOLS_ENABLED or not tools:
        # Fallback: classic single-shot
        text = robust_chat_completion(client, model, messages, temperature=temperature, **kwargs)
        return text, []

    if max_iterations is None:
        max_iterations = config.AGENT_MAX_TOOL_ITERATIONS

    conversation = list(messages)
    tool_call_log = []

    for iteration in range(max_iterations):
        response = None
        _empty_retry_done = False  # one empty-output retry per iteration

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    **kwargs,
                )

                # ── Empty-output guard ────────────────────────────────────────
                # If the model burned its token budget in a <think> block and
                # returned neither text nor tool calls, retry once before
                # treating it as a real error.
                _raw_check = response.choices[0].message.content or ""
                _has_tc = bool(getattr(response.choices[0].message, "tool_calls", None))
                _has_xml = "<tool_call>" in _raw_check
                if (
                    not strip_think_blocks(_raw_check).strip()
                    and not _has_tc
                    and not _has_xml
                    and not _empty_retry_done
                ):
                    _empty_retry_done = True
                    _finish = getattr(response.choices[0], "finish_reason", "unknown")
                    if log_callback:
                        log_callback(
                            f"   ⚠️ Leere Antwort vom Modell "
                            f"(finish_reason={_finish}). Einmal neu versuchen in 3s..."
                        )
                    time.sleep(3)
                    continue  # one free retry — does not count as a connection error

                break  # valid response (has content or tool calls)

            except (APIConnectionError, RateLimitError) as e:
                if attempt < 2:
                    wait = 3 * (attempt + 1)
                    if log_callback:
                        log_callback(f"   ⚠️ API-Verbindungsproblem: {e}. Neuer Versuch {attempt+2}/3 in {wait}s...")
                    time.sleep(wait)
                else:
                    return f"[FEHLER: Timeout/Connection nach 3 Versuchen: {e}]", tool_call_log
            except Exception as e:
                return f"[FEHLER: {e}]", tool_call_log

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        assistant_msg = choice.message

        # Clean content: remove <think> blocks and XML tool call remnants
        raw_content = assistant_msg.content or ""
        has_openai_tool_calls = bool(getattr(assistant_msg, "tool_calls", None))

        # ── Fallback: check XML-format tool calls (Qwen3-style) ────────────────
        xml_tool_calls = None
        if not has_openai_tool_calls:
            xml_tool_calls = _parse_xml_tool_calls(raw_content)

        # No tool call (neither OpenAI nor XML) → done
        if not has_openai_tool_calls and not xml_tool_calls:
            # Clean XML remnants and think blocks from final text
            text = _strip_xml_tool_calls(strip_think_blocks(raw_content))
            return text, tool_call_log

        hard_stop = False  # becomes true if a tool returns HARD_STOP_TOKEN

        # ── Process XML tool calls ─────────────────────────────────────────────
        if xml_tool_calls and not has_openai_tool_calls:
            # Synthetic assistant message without real tool_calls
            clean_text = _strip_xml_tool_calls(strip_think_blocks(raw_content))
            conversation.append({
                "role": "assistant",
                "content": raw_content,  # Keep original with XML for context
            })
            for xml_tc in xml_tool_calls:
                tool_name = xml_tc["name"]
                tool_args = xml_tc["args"]

                if log_callback:
                    log_callback(f"🔧 KI-Tool (XML): {tool_name}({_preview_args(tool_args)})")

                try:
                    result_str = str(tool_executor(tool_name, tool_args))
                    if not result_str.strip():
                        result_str = f"[Tool '{tool_name}' hat kein Ergebnis zurückgegeben]"
                except Exception as e:
                    result_str = (
                        f"[Tool-Fehler bei '{tool_name}': {type(e).__name__}: {e}. "
                        f"Bitte überprüfe die Argumente.]"
                    )

                if HARD_STOP_TOKEN in result_str:
                    hard_stop = True
                    result_str = result_str.replace(HARD_STOP_TOKEN, "").strip()

                tool_call_log.append({
                    "name": tool_name,
                    "args": tool_args,
                    "result_preview": result_str[:200],
                })
                conversation.append({
                    "role": "user",
                    "content": f"[Tool-Ergebnis für {tool_name}]: {result_str}",
                })

            if hard_stop:
                # Remove tools/tool_choice from kwargs — hard-stop should not enable another
                # tool-use loop, but deliver closing text directly.
                kwargs_clean = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
                last_text = robust_chat_completion(client, model, conversation,
                                                   temperature=temperature, **kwargs_clean)
                return last_text, tool_call_log
            continue  # Next iteration

        # ── Process OpenAI-format tool calls ───────────────────────────────────
        # Add assistant message with tool_calls to conversation
        conversation.append(assistant_msg)

        for tc in assistant_msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except Exception:
                tool_args = {}

            if log_callback:
                log_callback(f"🔧 KI-Tool: {tool_name}({_preview_args(tool_args)})")

            try:
                result_str = str(tool_executor(tool_name, tool_args))
                if not result_str.strip():
                    result_str = f"[Tool '{tool_name}' hat kein Ergebnis zurückgegeben]"
            except Exception as e:
                result_str = (
                    f"[Tool-Fehler bei '{tool_name}': {type(e).__name__}: {e}. "
                    f"Bitte überprüfe die Argumente oder wähle einen anderen Ansatz.]"
                )

            if HARD_STOP_TOKEN in result_str:
                hard_stop = True
                result_str = result_str.replace(HARD_STOP_TOKEN, "").strip()

            tool_call_log.append({
                "name": tool_name,
                "args": tool_args,
                "result_preview": result_str[:200],
            })

            # Send back tool response
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

        # Hard-stop: immediately generate closing text and finish.
        # Remove tools/tool_choice so no further tool-use loop occurs.
        if hard_stop:
            kwargs_clean = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
            last_text = robust_chat_completion(client, model, conversation,
                                               temperature=temperature, **kwargs_clean)
            return last_text, tool_call_log

    # Max iterations reached → return last available text
    print(f"   ⚠️ Agentic loop: Max Iterationen ({max_iterations}) erreicht.")
    last_text = robust_chat_completion(client, model, conversation, temperature=temperature, **kwargs)
    return _strip_xml_tool_calls(last_text), tool_call_log


# Compact representation of tool arguments for the log.
def _preview_args(args: dict) -> str:
    """Compact representation of tool arguments for the log."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        parts.append(f"{k}={v_str[:60]!r}" if len(v_str) > 60 else f"{k}={v_str!r}")
    return ", ".join(parts)


# ── Embedding ─────────────────────────────────────────────────────────────────

# Gets embedding for a text. Returns None on error.
def get_embedding(
    client: "OpenAI",
    text: str,
    model_id: str | None = None,
) -> list[float] | None:
    """Gets embedding for a text. Returns None on error."""
    if model_id is None:
        model_id = config.EMBEDDING_MODEL_ID
    try:
        text = text.replace("\n", " ").strip()
        if not text:
            return None
        return client.embeddings.create(input=[text], model=model_id).data[0].embedding
    except Exception:
        return None


# ── Think block removal ───────────────────────────────────────────────────────

# Removes <think>...</think> blocks from LLM responses (Qwen3 Extended Thinking).
def strip_think_blocks(text: str) -> str:
    """
    Removes <think>...</think> blocks from LLM responses (Qwen3 Extended Thinking).
    Applied to all LLM responses to prevent downstream parsing errors.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── XML tool call parser ───────────────────────────────────────────────────────

# Parses XML-format tool calls from content (Qwen3-style fallback).
def _parse_xml_tool_calls(content: str) -> list[dict] | None:
    """
    Parses XML-format tool calls from content (Qwen3-style fallback).
    Expects format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    Returns None if no XML tool calls found.
    """
    matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
    if not matches:
        return None

    calls = []
    for i, raw in enumerate(matches):
        try:
            data = json.loads(raw)
            name = data.get("name") or data.get("function")
            args = data.get("arguments") or data.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"id": f"xml_tc_{i}", "name": name, "args": args})
        except Exception:
            pass

    return calls if calls else None


# Removes <tool_call>...</tool_call> blocks from final text.
def _strip_xml_tool_calls(content: str) -> str:
    """Removes <tool_call>...</tool_call> blocks from final text."""
    return re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()


# ── JSON extraction ───────────────────────────────────────────────────────────

# Extracts JSON from LLM response. Tries multiple strategies.
def extract_json_robust(text: str):
    """
    Extracts JSON from LLM response. Tries multiple strategies.
    Returns None if nothing is parseable.
    Automatically removes <think> blocks before parsing.
    """
    text = strip_think_blocks(text).strip()

    def fix_and_parse(s: str):
        # Remove trailing commas
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return json.loads(s)

    # 1. ```json ... ``` code block
    if "```" in text:
        try:
            if "```json" in text:
                inner = text.split("```json", 1)[1].split("```", 1)[0]
            else:
                inner = text.split("```", 1)[1].split("```", 1)[0]
            return fix_and_parse(inner.strip())
        except Exception:
            pass

    # 2. Brute-force array [...]
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return fix_and_parse(text[start:end + 1])
    except Exception:
        pass

    # 3. Brute-force object {...}
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return fix_and_parse(text[start:end + 1])
    except Exception:
        pass

    # 4. Line repair: extract individual JSON objects {…} from text
    #    and combine as array. Catches cases where model outputs
    #    JSON objects line-by-line (without enclosing []).
    try:
        objects: list = []
        depth = 0
        buf = []
        for ch in text:
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}" and depth > 0:
                depth -= 1
                buf.append(ch)
                if depth == 0:
                    candidate = "".join(buf).strip()
                    try:
                        obj = fix_and_parse(candidate)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except Exception:
                        pass
                    buf = []
            elif depth > 0:
                buf.append(ch)
        if objects:
            return objects
    except Exception:
        pass

    return None
