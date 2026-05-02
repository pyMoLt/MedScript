# rag/tools.py — Tool definitions (OpenAI function calling schema) + ToolExecutor

# Tool definitions (OpenAI function calling schema) + ToolExecutor.
# Used by core/llm_client.agentic_chat_completion().

import time
import random
import config
from rag import retriever, web_search, url_fetcher


# ── Tool-Definitionen (OpenAI function calling Schema) ───────────────────────

TOOL_SEARCH_KNOWLEDGE_BASE = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Nutze dieses Tool, um medizinisches Faktenwissen aus den bereitgestellten Lehrbüchern abzufragen. "
            "Du kannst das Tool so oft nutzen, wie du möchtest, um verschiedene Begriffe zu recherchieren. "
            "REGEL: Wiederhole niemals exakt dieselbe Suchanfrage. "
            "Nutze die erhaltenen Antworten, um deinen Buchtext aufzubauen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchanfrage, z.B. 'Voltage-gated Kaliumkanal Repolarisation Inaktivierungsgate'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Anzahl zurückgegebener Textpassagen (1-5, Standard: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_INSERT_RAG_IMAGE = {
    "type": "function",
    "function": {
        "name": "insert_rag_image",
        "description": (
            "Fügt eine Lehrbuch-Abbildung aus dem RAG-Store in den Text ein. "
            "Bilder werden dir automatisch angeboten sobald search_knowledge_base "
            "passende Abbildungen im Lehrbuch findet. "
            "Rufe dieses Tool nach einer erfolgreichen search_knowledge_base-Suche auf, "
            "wenn der Hinweis '[BILDER-HINWEIS]' in der Antwort erschienen ist. "
            "Falls keines der angebotenen Bilder inhaltlich passt oder du kein Bild einfügen möchtest, "
            "übergib image_id='skip' — das Tool wird die Einfügung dann ohne Fehler überspringen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "ID der Abbildung aus der angebotenen Auswahlliste (z.B. 'rag_img_0')",
                },
                "caption": {
                    "type": "string",
                    "description": "Kurze deutsche Bildunterschrift (1-2 Sätze). Beschreibt was die Abbildung zeigt.",
                },
            },
            "required": ["image_id", "caption"],
        },
    },
}

TOOL_GET_STRUCTURED_COMPARISON = {
    "type": "function",
    "function": {
        "name": "get_structured_comparison",
        "description": (
            "Sucht nach vergleichenden Informationen für eine Tabellen-Darstellung. "
            "Ideal für: Differentialdiagnosen, Klassifikationssysteme, Medikamentenvergleiche, "
            "Stufenschemata, Symptom-Vergleiche zwischen Erkrankungen. "
            "Das Ergebnis wird als strukturierter Text zurückgegeben, "
            "den du in eine Markdown-Tabelle umwandeln kannst."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Vergleichsthema, z.B. 'Differentialdiagnose Dyspnoe kardial vs. pulmonal'"
                },
            },
            "required": ["topic"],
        },
    },
}

TOOL_SEARCH_WEB = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Durchsucht das Web via SearxNG nach medizinischen Informationen. "
            "Nutze dieses Tool zwingend, um Informationen zu verifizieren oder wenn die interne Knowledge Base "
            "keine ausreichende Detailtiefe liefert oder aktuellste Informationen (Leitlinien, neue Studien) nötig sind. "
            "Nutze dieses Tool auch dann, wenn search_knowledge_base keine oder unzureichende Ergebnisse liefert. "
            "REGEL: Wiederhole niemals exakt dieselbe Suchanfrage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchanfrage, z.B. 'Leitlinien 2024 Herzinsuffizienz Therapie ESC'"
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_FETCH_URL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Ruft den Textinhalt einer Webseite ab. "
            "Nutze dies wenn search_web eine relevante URL geliefert hat "
            "und du den vollständigen Inhalt lesen möchtest (z.B. Leitlinie, PubMed-Abstract, "
            "Fachgesellschafts-Empfehlung). "
            "Der Text wird auf ein Token-Limit gekürzt. "
            "WICHTIG: Nur öffentliche HTTP/HTTPS URLs. Keine lokalen Adressen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Vollständige URL der Webseite, z.B. 'https://www.escardio.org/...'"
                },
            },
            "required": ["url"],
        },
    },
}


# ── Tool-Liste aufbauen ───────────────────────────────────────────────────────

def get_tools_for_run(store_name: str | None, web_search_enabled: bool = None) -> list[dict]:
    """
    Gibt die Liste der Tool-Definitionen zurück die für diesen Lauf gelten.
    Reihenfolge signalisiert dem LLM die Priorität: RAG-Tools zuerst.

    HINWEIS: is_searxng_available() wird hier NICHT geprüft.
    Der live-Ping passiert in _exec_search_web beim tatsächlichen Aufruf.
    Das verhindert das Race-Condition-Problem wo SearxNG beim Start kurz nicht
    erreichbar ist und die Web-Tools dann für den ganzen Lauf fehlen.
    """
    if web_search_enabled is None:
        web_search_enabled = config.SEARXNG_ENABLED

    # Kein Tool-Use wenn weder RAG noch Websuche aktiv
    if store_name is None and not web_search_enabled:
        return []

    tools = []

    if store_name:
        tools.extend([
            TOOL_SEARCH_KNOWLEDGE_BASE,
            TOOL_INSERT_RAG_IMAGE,
            TOOL_GET_STRUCTURED_COMPARISON,
        ])

    # Websuche: nur wenn Nutzer aktiviert hat und global nicht deaktiviert
    # Erreichbarkeits-Check erfolgt beim tatsächlichen Tool-Call in _exec_search_web
    if web_search_enabled and config.SEARXNG_ENABLED:
        tools.append(TOOL_SEARCH_WEB)
        if config.URL_FETCH_ENABLED:
            tools.append(TOOL_FETCH_URL)

    return tools


# ── Tool-Executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Führt Tool-Calls aus. Wird an agentic_chat_completion() übergeben.
    Enthält Rate-Limiting für Websuche und URL-Fetch.
    """

    def __init__(self, store_name: str | None, log_callback: callable = None, model_id: str | None = None):
        self.store_name = store_name
        self.log_callback = log_callback
        # Dynamisches Modell: nutzt das aktuell geladene Modell des aufrufenden Modus.
        # Fallback auf TEXT_MODEL_SEARCH wenn kein Modell übergeben (z.B. in Tests).
        self.model_id: str = model_id if model_id else config.TEXT_MODEL_SEARCH
        # Global über den gesamten Lauf
        self.web_search_call_count: int = 0
        self.last_web_search_time: float = 0.0
        # Per Abschnitt — via reset_section_counters() zurücksetzen
        self.url_fetch_count_this_section: int = 0
        self.last_url_fetch_time: float = 0.0
        self.kb_search_cache: set[str] = set()
        # Bild-Puffer: sammelt Bilder aus parallelen Bild-Suchen in search_knowledge_base
        self._image_buffer: list[dict] = []   # {id, path, desc, score}
        self._image_insert_count: int = 0
        self._MAX_IMAGE_INSERTS: int = config.RAG_MAX_IMAGE_INSERTS_PER_SECTION

    def _log(self, msg: str) -> None:
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def execute(self, tool_name: str, args: dict, subagent_mode: bool = False) -> str:
        """Führt ein Tool aus und gibt das Ergebnis als String zurück."""
        try:
            if tool_name == "search_knowledge_base":
                return self._exec_search_knowledge_base(**args)
            elif tool_name == "insert_rag_image":
                return self._exec_insert_rag_image(**args)
            elif tool_name == "get_structured_comparison":
                return self._exec_get_structured_comparison(**args)
            elif tool_name == "search_web":
                # Im Subagenten-Modus führen wir nur die Roh-Suche ohne eigenen Agenten aus
                if subagent_mode:
                    return self._exec_search_web_raw(**args)
                return self._exec_search_web(**args)
            elif tool_name == "fetch_url":
                # Im Subagenten-Modus geben wir den Roh-Text zurück
                if subagent_mode:
                    return self._exec_fetch_url_raw(**args)
                return self._exec_fetch_url(**args)
            else:
                return f"Fehler: Unbekanntes Tool '{tool_name}'"
        except Exception as e:
            return f"Fehler bei Tool-Ausführung: {str(e)}"

    def _exec_search_web_raw(self, query: str) -> str:
        """Interne Roh-Suche für Subagenten (ohne Kompression).
        Rate-Limiting und globaler Call-Count gelten auch hier."""
        max_calls = config.SEARXNG_MAX_CALLS_PER_RUN
        if self.web_search_call_count >= max_calls:
            return f"Web-Suche-Limit erreicht ({max_calls}). Nutze nur vorhandene Informationen."

        # Jitter-Delay (identisch zu _exec_search_web)
        elapsed = time.time() - self.last_web_search_time
        jitter_delay = random.uniform(config.SEARXNG_JITTER_MIN, config.SEARXNG_JITTER_MAX)
        if elapsed < jitter_delay:
            wait = jitter_delay - elapsed
            self._log(f"⏳ Rate-Limit (Subagent-Roh): Warte {wait:.1f}s...")
            time.sleep(wait)

        self.web_search_call_count += 1
        self.last_web_search_time = time.time()
        self._log(f"🌐 Subagent-Suche: {query} (Aufruf {self.web_search_call_count}/{max_calls})")

        results = web_search.searxng_search(query, config.SEARXNG_RESULTS_PER_QUERY)
        return web_search.format_search_results_for_llm(results)

    def _exec_fetch_url_raw(self, url: str) -> str:
        """Interner Roh-Abruf für Subagenten (ohne Kompression)."""
        result = url_fetcher.fetch_url_content(url, config.URL_FETCH_MAX_CHARS)
        return url_fetcher.format_fetched_content_for_llm(result)

    def _exec_search_knowledge_base(self, query: str, max_results: int = 3) -> str:
        if not self.store_name:
            return "Kein RAG-Store konfiguriert."
            
        # Stop-Schild Methode: Exakt dieselbe Suche verhindern
        query_normalized = query.strip().lower()
        if query_normalized in self.kb_search_cache:
            self._log(f"   🛑 Blockiert: Identische RAG-Suche ({query})")
            return (
                f"[FEHLER: Du hast die exakt selbe Suchanfrage '{query}' bereits gestellt! "
                "Die Ergebnisse befinden sich bereits weiter oben in deinem Chatverlauf. "
                "Bitte frage nicht nochmal das Gleiche. Nutze die vorhandenen Informationen "
                "und schreibe jetzt den Text weiter.]"
            )
        self.kb_search_cache.add(query_normalized)
            
        self._log(f"🔍 RAG-Suche: {query}")
        results = retriever.retrieve(query, self.store_name, top_k=max_results)
        
        import json
        if not results:
            self._log(f"   → 0 Treffer")
            return json.dumps({
                "status": "error",
                "extracted_data": "Keine relevanten Ergebnisse in der Datenbank gefunden.",
                "system_directive": "Suche fehlgeschlagen. Nutze dein eigenes Wissen oder versuche eine ANDERE, alternative Suchanfrage."
            }, ensure_ascii=False)
            
        self._log(f"   → {len(results)} Treffer")
        
        # Agentic Retrieval: Subagent für Context Compression
        raw_parts = []
        for i, r in enumerate(results, 1):
            raw_parts.append(f"Quelle: {r['source']}\nText: {r['text']}")
        raw_text = "\n\n".join(raw_parts)

        subagent_prompt = (
            f"Du bist ein Recherche-Assistent. Lies die folgenden Rohtexte und fasse NUR die Fakten zusammen, "
            f"die die Suchanfrage '{query}' beantworten.\n"
            "Schreibe kurz und bündig. Erfinde nichts dazu. Wenn die Information in den Texten "
            "nicht enthalten ist, schreibe 'Information nicht gefunden'.\n\n"
            f"ROHTEXTE:\n{raw_text}"
        )

        from core.llm_client import create_openai_client, robust_chat_completion
        client = create_openai_client()
        
        self._log(f"   🧠 Subagent verdichtet RAG-Texte...")
        compressed_text = robust_chat_completion(
            client=client,
            model=self.model_id,
            messages=[{"role": "user", "content": subagent_prompt}],
            temperature=config.WRITING_TEMPERATURE,
            max_tokens=config.KB_SUBAGENT_MAX_TOKENS
        )

        result_json = {
            "status": "success",
            "extracted_data": compressed_text,
            "system_directive": "Information erfolgreich verarbeitet. Du kannst nun den Text schreiben oder bei Bedarf nach weiteren, ANDEREN Begriffen suchen."
        }

        # ── Parallele Bild-Suche (ohne Subagent, direkt) ─────────────────────
        # Läuft nach dem Subagenten und nutzt dieselbe Query mit Bild-Präfix.
        # Ergebnisse landen im Puffer — die KI kann sie via insert_rag_image abrufen.
        try:
            image_query = f"Abbildung Schema Foto Bild Diagramm Illustration {query}"
            img_raw = retriever.retrieve(
                image_query, self.store_name,
                top_k=3,
                min_score=config.RAG_IMAGE_MIN_SCORE,
                include_images=True,
            )
            img_hits = [
                r for r in img_raw
                if r.get("type") == "image_description" and r.get("image_path")
            ]
            new_count = 0
            for r in img_hits:
                path = r["image_path"]
                # Duplikat-Check: gleichen Pfad nicht zweimal in Puffer
                if not any(b["path"] == path for b in self._image_buffer):
                    img_id = f"rag_img_{len(self._image_buffer)}"
                    self._image_buffer.append({
                        "id": img_id,
                        "path": path,
                        "desc": r["text"][:200],
                        "score": r["score"],
                    })
                    new_count += 1
            if new_count > 0:
                self._log(f"   🖼️ {new_count} RAG-Bild(er) im Puffer ({len(self._image_buffer)} gesamt, Score ≥ {config.RAG_IMAGE_MIN_SCORE})")
                result_json["image_hint"] = (
                    f"[BILDER-HINWEIS] Es wurden {new_count} neue Lehrbuch-Abbildung(en) zu diesem Thema gefunden. "
                    f"Insgesamt {len(self._image_buffer)} Bild(er) verfügbar. "
                    f"Nutze das Tool 'insert_rag_image' wenn eine Abbildung den Text bereichern würde."
                )
        except Exception as e:
            self._log(f"   ⚠️ Parallele Bild-Suche fehlgeschlagen (unkritisch): {e}")

        return json.dumps(result_json, ensure_ascii=False)

    def _exec_insert_rag_image(self, image_id: str, caption: str = "") -> str:
        """
        Fügt ein Bild aus dem Puffer in den Text ein.
        Validiert image_id gegen den internen Puffer (kein Halluzinationsrisiko).
        Sonderfall: image_id='none' oder 'skip' → explizites Ablehnen ohne Fehler.
        """
        import json as _json
        from pathlib import Path as _Path

        # Explizites Ablehnen durch die KI — kein Bild passt
        if image_id.lower().strip() in ("none", "skip", "kein", "keines"):
            self._log("   🖼️ KI hat Bild-Einfügung abgelehnt (kein passendes Bild)")
            return "[Verstanden: Kein Bild wird eingefügt. Schreibe den Text ohne Abbildung weiter.]"

        if not self._image_buffer:
            return (
                "[Noch keine RAG-Bilder verfügbar. Bilder werden automatisch angeboten wenn "
                "search_knowledge_base passende Lehrbuch-Abbildungen findet. "
                "Schreibe den Text ohne Bild weiter. "
                "Falls du kein Bild einfügen möchtest, übergib image_id='skip'.]"
            )

        if self._image_insert_count >= self._MAX_IMAGE_INSERTS:
            return (
                f"[Maximum von {self._MAX_IMAGE_INSERTS} RAG-Bildern pro Abschnitt erreicht. "
                "Fahre ohne weiteres Bild fort.]"
            )

        available = {b["id"]: b for b in self._image_buffer}
        if image_id not in available:
            buffer_list = "\n".join(
                f"  \u2022 {b['id']} (Score {b['score']}): {b['desc'][:100]}..."
                for b in self._image_buffer
            )
            return (
                f"[Ungültige Bild-ID '{image_id}'.\n"
                f"Verfügbare Bilder:\n{buffer_list}\n"
                f"Wähle eine ID aus der Liste oder übergib image_id='skip' wenn keins passt.]"
            )

        img = available[image_id]
        src_path = _Path(img["path"])
        if not src_path.exists():
            return f"[Bilddatei nicht gefunden: {img['path']}]"

        self._image_insert_count += 1
        caption_text = caption.strip() or "Lehrbuch-Abbildung"
        md_tag = f"[[RAG_IMAGE:{img['path']}]]"
        self._log(f"   🖼️ RAG-Bild eingefügt: {image_id} ({src_path.name}) — {caption_text[:60]}")

        return (
            f"[ERFOLG: Bild '{image_id}' bereit. "
            f"WICHTIG: Füge diesen Tag an der passenden Stelle in deinen Fließtext ein:\n"
            f"{md_tag}\n*{caption_text}*]"
        )

    def _exec_get_structured_comparison(self, topic: str) -> str:
        """
        Erstellt einen strukturierten Vergleich / eine Tabelle zu einem medizinischen Thema.
        Ablauf: RAG-Suche → LLM-Subagent formatiert die Daten → Markdown-Tabelle zurück.
        """
        import json as _json
        if not self.store_name:
            return "[get_structured_comparison: Kein RAG-Store konfiguriert.]"

        self._log(f"📊 Strukturierter Vergleich: {topic}")

        # 1. RAG-Suche nach relevanten Chunk-Daten
        query = f"{topic} Vergleich Tabelle Klassifikation Übersicht Kriterien"
        results = retriever.retrieve(query, self.store_name, top_k=5)
        if not results:
            # Kein RAG-Material → LLM aus eigenem Wissen
            raw_context = ""
        else:
            raw_context = "\n\n---\n\n".join(
                f"[Quelle: {r['source']}, Score: {r['score']}]\n{r['text']}"
                for r in results
            )

        # 2. LLM-Subagent formatiert die Daten als Tabelle
        if raw_context:
            subagent_prompt = (
                f"Du bist ein medizinischer Lernassistent. Erstelle eine übersichtliche "
                f"Markdown-Tabelle oder strukturierte Liste zum Thema: **{topic}**\n\n"
                f"Nutze dazu folgende Informationen aus dem Lehrbuch:\n\n"
                f"{raw_context[:3000]}\n\n"
                f"ANFORDERUNGEN:\n"
                f"- Erstelle eine klare Markdown-Tabelle (| Spalte | ... |) oder \n"
                f"  eine nummerierte/gegliederte Liste — je nach Thema was sinnvoller ist.\n"
                f"- Fokus: Vergleiche, Klassifikationen, Kriterien, Stufenschemata.\n"
                f"- Maximal 400 Wörter. Kein Fließtext — nur Tabelle/Struktur.\n"
                f"- Sprache: Deutsch. Keine Einleitung, direkt mit der Tabelle beginnen."
            )
        else:
            subagent_prompt = (
                f"Du bist ein medizinischer Lernassistent. Erstelle eine übersichtliche "
                f"Markdown-Tabelle oder strukturierte Liste zum Thema: **{topic}**\n\n"
                f"Nutze dein medizinisches Fachwissen.\n"
                f"ANFORDERUNGEN:\n"
                f"- Klare Markdown-Tabelle oder nummerierte Liste.\n"
                f"- Maximal 400 Wörter. Kein Fließtext — nur Tabelle/Struktur.\n"
                f"- Sprache: Deutsch. Direkt mit der Tabelle beginnen."
            )

        try:
            from core.llm_client import create_openai_client, robust_chat_completion
            client = create_openai_client()
            table_text = robust_chat_completion(
                client=client,
                model=self.model_id,
                messages=[{"role": "user", "content": subagent_prompt}],
                temperature=0.1,
                max_tokens=config.ANALYSIS_MAX_TOKENS,
            )
            if not table_text or not table_text.strip():
                table_text = raw_context[:1500] if raw_context else "Keine Daten verfügbar."
        except Exception as e:
            self._log(f"   ⚠️ Subagent-Fehler bei get_structured_comparison: {e}")
            table_text = raw_context[:1500] if raw_context else "Subagent-Fehler."

        self._log(f"   ✅ Vergleichstabelle erstellt ({len(table_text)} Zeichen)")

        result = {
            "status": "success",
            "topic": topic,
            "structured_comparison": table_text.strip(),
            "system_directive": (
                "Vergleichstabelle erfolgreich erstellt. "
                "Integriere sie sinnvoll in den Fließtext — direkt einbetten oder als Ergänzung referenzieren."
            ),
        }
        return _json.dumps(result, ensure_ascii=False)

    def _exec_search_web(self, query: str) -> str:
        max_calls = config.SEARXNG_MAX_CALLS_PER_RUN
        if self.web_search_call_count >= max_calls:
            return f"Web-Suche-Limit erreicht ({max_calls}). Nutze nur Lehrbuch-Wissen."

        # Erreichbarkeits-Check beim tatsächlichen Aufruf (nicht beim Tool-Aufbau)
        if not web_search.is_searxng_available():
            self._log("⚠️ SearxNG nicht erreichbar — Websuche übersprungen")
            return (
                "[Websuche nicht verfügbar: SearxNG-Server antwortet nicht. "
                "Nutze stattdessen search_knowledge_base oder schreibe aus eigenem Wissen weiter.]"
            )

        # Rate-Limiting mit Jitter (4-8 Sekunden variabel)
        elapsed = time.time() - self.last_web_search_time
        jitter_delay = random.uniform(config.SEARXNG_JITTER_MIN, config.SEARXNG_JITTER_MAX)
        if elapsed < jitter_delay:
            wait = jitter_delay - elapsed
            self._log(f"⏳ Rate-Limit (Jitter): Warte {wait:.1f}s...")
            time.sleep(wait)

        self.web_search_call_count += 1
        self.last_web_search_time = time.time()
        self._log(f"🌐 Websuche: {query} (Aufruf {self.web_search_call_count}/{max_calls})")

        results = web_search.searxng_search(query, config.SEARXNG_RESULTS_PER_QUERY)
        import json
        
        raw_text = web_search.format_search_results_for_llm(results)
        
        if "Fehler" in raw_text or not results:
            return json.dumps({
                "status": "error",
                "extracted_data": raw_text,
                "system_directive": "Websuche fehlgeschlagen oder keine Ergebnisse."
            }, ensure_ascii=False)

        subagent_prompt = (
            f"Du bist ein Medizin-Recherche-Assistent. Beantworte die Suchanfrage '{query}' "
            "mit präzisen, belegten Fakten aus aktuellen Quellen.\n\n"
            "PFLICHT-REGELN:\n"
            "1. 'fetch_url': Öffne JEDE vielversprechende URL (Leitlinie, PubMed, Fachgesellschaft) zwingend "
            "mit fetch_url — Snippets allein reichen nie für medizinische Fakten.\n"
            "2. 'search_web': Falls die vorliegenden Ergebnisse unzureichend, veraltet oder zu allgemein sind, "
            "starte SOFORT eine verfeinerte Folgesuche. Nutze spezifischere Suchbegriffe "
            "(z.B. Leitlinienname, ICD-Code, Wirkstoffname). Wiederhole bis du belastbare Fakten hast "
            "oder die maximale Suchanzahl erreicht ist.\n"
            "3. Wenn keine nützlichen Informationen gefunden wurden: Gib explizit zurück "
            "'Keine ausreichenden Informationen gefunden.' — erfinde NICHTS.\n\n"
            "ZIEL: Aktuelle klinische Fakten (Leitlinien, Studiendaten, Dosierungen, Empfehlungsgrade). "
            "Fasse NUR gesicherte Informationen zusammen, mit Quellenangabe wenn möglich.\n\n"
            f"SUCHERGEBNISSE (Ausgangspunkt):\n{raw_text}"
        )

        # Definition der Subagenten-Tools (interne Verlinkung zu Raw-Executors)
        sub_tools = [TOOL_FETCH_URL, TOOL_SEARCH_WEB]

        from core.llm_client import create_openai_client, agentic_chat_completion
        client = create_openai_client()
        
        self._log(f"   🧠 Recherche-Assistent (Subagent) beginnt Web-Analyse...")
        
        # Pass subagent_mode=True so the subagent receives raw data instead of
        # recursively spawning further subagents. max_tokens is required here —
        # without it the model has no token cap and can get stuck in a thinking
        # loop indefinitely (the main WRITING_MAX_TOKENS limit does NOT apply).
        compressed_text, _ = agentic_chat_completion(
            client=client,
            model=self.model_id,
            messages=[{"role": "user", "content": subagent_prompt}],
            tools=sub_tools,
            tool_executor=lambda name, args: self.execute(name, args, subagent_mode=True),
            log_callback=None,
            max_iterations=config.WEB_SUBAGENT_MAX_ITERATIONS,
            max_tokens=config.WEB_SUBAGENT_MAX_TOKENS,
        )

        failed = compressed_text.startswith("[FEHLER")
        result_json = {
            "status": "error" if failed else "success",
            "extracted_data": compressed_text,
            "system_directive": (
                "Web-Suche fehlgeschlagen. Bitte ohne diese Information fortfahren."
                if failed else
                "Web-Information erfolgreich verarbeitet."
            ),
        }
        return json.dumps(result_json, ensure_ascii=False)

    def _exec_fetch_url(self, url: str) -> str:
        max_per_section = config.URL_FETCH_MAX_CALLS_PER_SECTION
        if self.url_fetch_count_this_section >= max_per_section:
            return f"URL-Fetch-Limit für diesen Abschnitt erreicht ({max_per_section})."

        # Rate-Limiting
        elapsed = time.time() - self.last_url_fetch_time
        if elapsed < config.URL_FETCH_MIN_DELAY_SECONDS:
            time.sleep(config.URL_FETCH_MIN_DELAY_SECONDS - elapsed)

        result = url_fetcher.fetch_url_content(url, config.URL_FETCH_MAX_CHARS)
        self.url_fetch_count_this_section += 1
        self.last_url_fetch_time = time.time()

        chars = len(result.get("text", ""))
        self._log(f"🔗 URL abrufen: {url[:80]}... ({chars} Zeichen extrahiert)")
        raw_text = url_fetcher.format_fetched_content_for_llm(result)

        import json
        if "Fehler" in raw_text:
            return json.dumps({
                "status": "error",
                "extracted_data": raw_text,
                "system_directive": "URL konnte nicht gelesen werden."
            }, ensure_ascii=False)

        subagent_prompt = (
            f"Du bist ein Recherche-Assistent. Fasse die Kerninformationen des folgenden Webseiten-Textes zusammen, "
            f"die für das Schreiben eines medizinischen Fachtextes relevant sein könnten.\n"
            "Schreibe kurz und bündig. Erfinde nichts dazu.\n\n"
            f"WEBSEITEN-TEXT:\n{raw_text}"
        )

        from core.llm_client import create_openai_client, agentic_chat_completion
        client = create_openai_client()
        
        self._log(f"   🧠 Lese-Assistent (Subagent) analysiert Webseite...")
        
        # max_tokens is required — without it the subagent has no token cap
        # and can exhaust the context in a thinking loop.
        compressed_text, _ = agentic_chat_completion(
            client=client,
            model=self.model_id,
            messages=[{"role": "user", "content": subagent_prompt}],
            tools=[],  # fetch_url subagent needs no further tools
            tool_executor=lambda name, args: self.execute(name, args, subagent_mode=True),
            log_callback=None,
            max_iterations=config.FETCH_SUBAGENT_MAX_ITERATIONS,
            max_tokens=config.FETCH_SUBAGENT_MAX_TOKENS,
        )

        failed = compressed_text.startswith("[FEHLER")
        result_json = {
            "status": "error" if failed else "success",
            "extracted_data": compressed_text,
            "system_directive": (
                "URL-Fetch fehlgeschlagen. Bitte ohne diese Information fortfahren."
                if failed else
                "Webseite erfolgreich gelesen und verarbeitet."
            ),
        }
        return json.dumps(result_json, ensure_ascii=False)

    def reset_section_counters(self) -> None:
        """Setzt per-Sektion-Zähler zurück. Vor jedem neuen Kapitel aufrufen."""
        self.url_fetch_count_this_section = 0
        self.kb_search_cache.clear()
        # Bild-Puffer und Insert-Zähler pro Abschnitt zurücksetzen
        self._image_buffer.clear()
        self._image_insert_count = 0
