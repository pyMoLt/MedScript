# modes/summary.py — Mode 1: single file to detailed summary

# Mode 1: Single file → detailed summary (refactored from pdf_to_detailed_Summary.py)

from __future__ import annotations
from pathlib import Path
import re
import threading
import markdown

import config
from core.llm_client import (
    create_openai_client, robust_chat_completion, agentic_chat_completion,
    switch_model, unload_model,
)
from core.pdf_parser import (
    repair_pdf_if_needed, build_docling_converter, ingest_pdf,
    get_item_context, is_image_in_margin,
)
from core.image_utils import AdvancedDeduplicator, pil_to_base64_jpeg, save_pil_image, get_image_hash
from core.text_utils import (
    is_text_junk, smart_split_text, clean_llm_markdown_output,
    inject_markdown_headers, extract_last_n_chars, extract_first_n_chars,
)
from core.output import assemble_final_markdown, markdown_to_pdf, save_markdown
from core.cache import SimpleCache
from core.page_renderer import (
    get_text_writing_cache_key, load_text_writing_cache, save_text_writing_cache,
)
from rag.tools import get_tools_for_run, ToolExecutor
from rag.augmenter import get_writer_tool_instruction, resolve_rag_image_tags


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_post_process_prompt(detail_level: int, tools_enabled: bool = False) -> str:
    """Erstellt den System-Prompt für Text-Post-Processing. Drei Detailstufen (portiert + erweitert)."""
    base = (
        "Du bist ein Lektor für medizinische Skripte mit höchstem akademischen Anspruch. "
        "Du erhältst einen Text-Abschnitt zur Bearbeitung sowie den Kontext davor/danach.\n\n"
        "DEINE AUFGABEN:\n"
        "1. FOKUS: Bearbeite NUR den mittleren Teil 'TEXT ZUR BEARBEITUNG'.\n"
        "2. SPRACHE:\n"
        "   - Korrigiere Grammatik- und OCR-Fehler.\n"
        "3. STRUKTUR & LAYOUT (SEHR WICHTIG):\n"
        "   - FORMATIERUNG: Nutze **Fettungen** für wichtige Fachbegriffe, Medikamente oder Schlüsselkonzepte.\n"
        "   - LISTEN: Nutze Bullet-Points für Aufzählungen oder Schritte, um Bleiwüsten zu vermeiden.\n"
        "   - ABSÄTZE: Mache spätestens alle 5-6 Sätze einen sinnvollen Absatz.\n"
        "   - ÜBERSCHRIFTEN: Wenn du erkennst, dass ein neuer Abschnitt beginnt "
        "(und keine Markdown-Überschrift vorhanden ist), füge sinnvolle '### Überschriften' ein. "
        "WICHTIG: Überschriften dürfen niemals nummeriert werden oder Präfixe wie 'Abschnitt' enthalten. Beginne direkt mit dem Fachbegriff!\n"
        "   - MERKSÄTZE: Wenn etwas besonders wichtig ist, nutze einen Zitat-Block (> Merksatz: ...).\n"
    )

    if detail_level >= 90:
        instruction = (
            "4. DETAILS (INHALT \u2014 100% TREUE):\n"
            "   - STIL: Schreibe in vollständigen, akademischen Sätzen. Wandle Stichpunkte in Fließtext um.\n"
            "   - Behalte JEDES fachliche Detail. Entferne keine Zahlen, Dosen oder Mechanismen.\n"
            "   - Der Text muss ein Lehrbuch ersetzen können. Maximale Ausführlichkeit bei perfekter Lesbarkeit.\n"
        )
    elif detail_level >= 40:
        instruction = (
            "4. DETAILS (INHALT \u2014 ZUSAMMENFASSUNG):\n"
            "   - STIL: Gut lesbarer Fließtext, strukturiert durch Absätze.\n"
            "   - Behalte ALLE Informationen, aber fasse sie kompakter zusammen.\n"
            "   - Entferne redundante Füllwörter, aber behalte alle Fakten, Zahlen und Dosen.\n"
            "   - Fachlich korrekt und gut strukturiert.\n"
        )
    else:
        instruction = (
            "4. DETAILS (INHALT \u2014 CHEAT-SHEET):\n"
            "   - STIL: Nutze primär Stichpunkte (Bulletpoints). Fließtext nur für kurze Zusammenfassungen.\n"
            "   - REDUZIERE radikal auf das Wesentliche.\n"
            "   - Entferne ausschweifende Erklärungen. Behalte NUR harte Fakten, Schlüsselbegriffe, Zahlen und Dosen.\n"
        )

    constraints = "   - MATHE: Formeln müssen in Sprache überführt werden (z.B. Bruch als \"A geteilt durch B\"). Kein LaTeX.\n"

    if tools_enabled:
        closer = (
            "   - Wenn Details fehlen oder unklar sind, NUTZE DEINE TOOLS um Lehrbuchwissen abzufragen.\n"
            "   - Erfinde NICHTS, nutze Tools für Fakten. "
            "Antworte mit dem bearbeiteten Text (inkl. Tool-Ergänzungen)."
        )
    else:
        closer = "   - Erfinde NICHTS. Antworte AUSSCHLIESSLICH mit dem bearbeiteten Haupttext."

    return base + instruction + constraints + closer


def get_image_prompt(detail_level: int, context_str: str) -> str:
    """Erstellt den Prompt für Bild-Analyse je nach Detail-Level (portiert + erweitert)."""
    base_role = (
        "Du bist ein hochspezialisierter wissenschaftlicher Assistent für Medizin und Biologie. "
        "Deine Aufgabe ist die Analyse einer Vorlesungsgrafik.\n\n"
        "KONTEXT-INFO (Text um das Bild):\n"
        f"\"\"\"{context_str[:800]}\"\"\"\n"
        "NUTZE DIESEN KONTEXT ZUR ORIENTIERUNG! Er hilft dir, das Thema einzuordnen.\n"
        "WICHTIG: Dein Fokus liegt auf dem BILD. Beschreibe, was TATSÄCHLICH zu sehen ist.\n"
        "Nutze den Kontext, um die visuellen Elemente zu deuten, aber wiederhole ihn nicht einfach. "
        "Der Mehrwert deiner Analyse sind die Details (Pfeile, Beschriftungen, Schritte), "
        "die im Fließtext fehlen.\n\n"
    )

    if detail_level >= 80:
        instructions = (
            "ANALYSE-MODUS (MAXIMALE GENAUIGKEIT):\n"
            "1. FILTERUNG:\n"
            "   - Müll/Deko/leere Seite \u2192 'SKIP'\n"
            "   - Einfaches Foto ohne Lehrinhalt \u2192 'KEEP_SIMPLE'\n"
            "   - WISSEN (Diagramm, Mechanismus, Histologie, Grafik mit Labels) \u2192 'KEEP_ANALYSIS'\n\n"
            "2. TIEFENANALYSE (Für 'KEEP_ANALYSIS'):\n"
            "   - FORMATIERUNG: Nutze **Fettungen** für wichtige Begriffe. Nutze Listen (-) für Prozesse.\n"
            "   - MECHANISMEN: Beschreibe JEDEN Schritt im Detail (Moleküle, Rezeptoren, Signalwege).\n"
            "   - DIAGRAMME: Nenne konkrete Werte, Trends und Vergleiche.\n"
            "   - TEXT IM BILD: Transkribiere wichtige Labels, die nicht im Kontext stehen.\n"
            "   - Anatomie/Histologie: Erkläre Pathologien und sichtbare Strukturen genau.\n"
            "   - UMFANG: Sei ausführlich! Max 800 Tokens. Erkläre es so, dass man das Bild vor Augen hat.\n\n"
            "ANTWORT-FORMAT (genau eines wählen):\n"
            "- 'SKIP'\n"
            "- 'KEEP_SIMPLE: [Prägnanter Titel]'\n"
            "- 'KEEP_ANALYSIS: **Kernaussage:** [Fazit]. [Detaillierte, schrittweise Erklärung "
            "der Mechanismen und Daten mit **Fettungen** und Listen].'"
        )
    elif detail_level >= 40:
        instructions = (
            "ANALYSE-MODUS (ZUSAMMENFASSUNG):\n"
            "1. FILTERUNG:\n"
            "   - Unwichtiges, Deko \u2192 'SKIP'\n"
            "   - Wichtiges Wissen \u2192 'KEEP_ANALYSIS'\n\n"
            "2. ANALYSE (Für 'KEEP_ANALYSIS'):\n"
            "   - Fasse die KERNAUSSAGE zusammen.\n"
            "   - Strukturiere deine Antwort sauber. Nutze **Fettungen** für wichtige Begriffe.\n"
            "   - Ignoriere kleinste Details, nenne nur den allgemeinen Trend (steigt/fällt).\n"
            "   - Beschreibe Mechanismen grob (Input \u2192 Output), ohne jedes Zwischenmolekül.\n"
            "   - UMFANG: Kompakt. Max 5-8 Sätze.\n\n"
            "ANTWORT-FORMAT:\n"
            "- 'SKIP'\n"
            "- 'KEEP_ANALYSIS: **Zusammenfassung:** [Erklärung der Hauptaussage].'"
        )
    else:
        instructions = (
            "ANALYSE-MODUS (CHEAT-SHEET / MINIMAL):\n"
            "1. FILTERUNG: Sei STRENG. Behalte das Bild NUR, wenn es absolut essenziell für das Verständnis ist. "
            "Alles andere \u2192 'SKIP'.\n\n"
            "2. ANALYSE:\n"
            "   - Keine langen Erklärungen.\n"
            "   - Gib nur eine Bildunterschrift, die sagt, was zu sehen ist.\n\n"
            "ANTWORT-FORMAT:\n"
            "- 'SKIP'\n"
            "- 'KEEP_SIMPLE: [Kurzer Titel/Unterschrift]'"
        )

    return base_role + instructions


TABLE_PROMPT = (
    "Du analysierst eine Tabelle aus einer medizinischen Vorlesung.\n"
    "Erstelle eine vollständige Markdown-Tabelle mit allen Daten.\n"
    "Behalte alle Beschriftungen und Werte bei. Keine Erklärungen, nur die Tabelle."
)


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────

def process_single_file(
    source_path: Path,
    settings: dict,
    progress_callback: callable = None,
    log_callback: callable = None,
    cancel_event: threading.Event = None,
    preview_callback: callable = None,
) -> dict:
    """
    Hauptfunktion für Einzel-Datei Verarbeitung.
    Gibt zurück: {'success': bool, 'output_path': Path, 'markdown': str, 'error': str}
    """

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            log("⚠️ Abbruch bestätigt.")
            raise InterruptedError("Vorgang durch Benutzer abgebrochen.")

    detail_level = settings.get("detail_level", 100)
    do_post = settings.get("do_post_processing", True)
    rag_store_name = settings.get("rag_store_name", None)
    output_format = settings.get("output_format", "pdf")
    web_enabled = settings.get("web_search_enabled", False)
    project_name = settings.get("project_name", None)

    # WICHTIG: Einstellungen neu laden
    from core.settings_manager import get_settings
    get_settings()

    try:
        client = create_openai_client()
        output_dir = Path(config.CENTRAL_OUTPUT_DIR)  # Path() absichern falls str aus settings.json
        stem = project_name or source_path.stem
        output_dir = output_dir / stem
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        cache = SimpleCache(output_dir / ".cache.json")

        # ── PHASE A: INGESTION ────────────────────────────────────────────────
        check_cancel()
        log(f"📄 Starte Ingestion: {source_path.name}")
        progress(5, "PDF wird repariert...")
        working_path = repair_pdf_if_needed(source_path)

        progress(8, "Docling konvertiert PDF...")
        try:
            converter = build_docling_converter()
            all_items, doc = ingest_pdf(working_path, converter)
        except Exception as e:
            return {"success": False, "output_path": None, "markdown": "", "error": str(e)}
        finally:
            if working_path != source_path and working_path.exists():
                try:
                    working_path.unlink()
                except Exception:
                    pass

        log(f"✅ Docling: {len(list(all_items))} Elemente")

        # Registry aufbauen
        try:
            from docling_core.types.doc import TextItem, TableItem, PictureItem
        except ImportError:
            from docling.datamodel.base_models import TextItem, TableItem, PictureItem  # type: ignore

        registry = []  # [{'type': ..., 'content': ..., 'image': ..., 'context': ..., 'processed': None}]
        dedup = AdvancedDeduplicator()

        all_items_list = list(all_items)
        progress(12, "Registry aufbauen...")
        # te_count verfolgt die Anzahl der bisher gesammelten Text-Einträge.
        # Bilder erhalten text_idx_before=te_count, damit sie bei der Assembly
        # inline zwischen den richtigen Text-Chunks platziert werden können.
        te_count = 0
        for item_idx, (item, level) in enumerate(all_items_list):
            if item_idx % 20 == 0:
                check_cancel()
            if isinstance(item, TextItem):
                text = item.text.strip()
                label = str(getattr(item, "label", "")).lower()
                if is_text_junk(text, label):
                    continue
                text = inject_markdown_headers(text, label)
                registry.append({"type": "text", "content": text, "processed": None})
                te_count += 1

            elif isinstance(item, TableItem):
                try:
                    pil_img = item.get_image(doc)
                    if pil_img:
                        registry.append({
                            "type": "table",
                            "content": "",
                            "image": pil_img,
                            "context": "",
                            "processed": None,
                            "text_idx_before": te_count,
                        })
                except Exception:
                    pass

            elif isinstance(item, PictureItem):
                try:
                    pil_img = item.get_image(doc)
                    if pil_img is None:
                        continue
                    w, h = pil_img.size
                    if min(w, h) < config.MIN_PIXEL_SIDE:
                        continue
                    if is_image_in_margin(item, doc):
                        continue
                    img_hash = get_image_hash(pil_img)
                    if dedup.is_duplicate(pil_img, img_hash):
                        continue
                    ctx = get_item_context(all_items_list, item_idx)
                    registry.append({
                        "type": "image",
                        "content": "",
                        "image": pil_img,
                        "context": ctx,
                        "processed": None,
                        "text_idx_before": te_count,
                    })
                except Exception as e:
                    log(f"   ⚠️ Bild-Extraktion: {e}")

        log(f"   Registry: {sum(1 for r in registry if r['type']=='text')} Text, "
            f"{sum(1 for r in registry if r['type']=='image')} Bilder, "
            f"{sum(1 for r in registry if r['type']=='table')} Tabellen")

        # Text-Chunks aufteilen
        text_entries = [r for r in registry if r["type"] == "text"]
        combined_text = "\n\n".join(r["content"] for r in text_entries)
        chunks = smart_split_text(combined_text)
        log(f"   {len(chunks)} Text-Chunks")

        # Positions-Mapping: text_entry_index → chunk_index
        # Wird in Phase D genutzt um Bilder inline zwischen Chunks einzufügen.
        _te_offsets: list[int] = []
        _off = 0
        for r in text_entries:
            _te_offsets.append(_off)
            _off += len(r["content"]) + 2  # +2 für "\n\n"-Trenner

        _chunk_end_offsets: list[int] = []
        _pos = 0
        for _chunk in chunks:
            _pos += len(_chunk) + 2
            _chunk_end_offsets.append(_pos)

        def _te_to_chunk(j: int) -> int:
            """Gibt den Chunk-Index zurück, der Text-Entry j enthält."""
            if not _chunk_end_offsets or j < 0:
                return -1
            if j >= len(_te_offsets):
                return len(_chunk_end_offsets) - 1
            te_off = _te_offsets[j]
            for ci, end in enumerate(_chunk_end_offsets):
                if te_off < end:
                    return ci
            return len(_chunk_end_offsets) - 1

        # ── PHASE B: TEXT PROCESSING ──────────────────────────────────────────
        processed_texts = []
        model_id = None  # wird in Phase B gesetzt, in Phase B2 (Titel) genutzt
        if do_post and chunks:
            progress(20, "Text-Modell lädt...")
            log(f"🔄 Lade Text-Modell...")
            t_mod = settings.get("text_model", config.TEXT_MODEL_LOAD)
            model_id = switch_model(client, t_mod, t_mod)

            tools = get_tools_for_run(rag_store_name, web_enabled)
            executor = ToolExecutor(rag_store_name, log_callback, model_id=model_id) if tools else None
            tools_enabled = bool(rag_store_name or web_enabled)
            system_prompt = get_post_process_prompt(detail_level, tools_enabled)

            if rag_store_name:
                tool_instr = get_writer_tool_instruction(rag_store_name, web_enabled)
                system_prompt += "\n\n" + tool_instr

            covered_headers: list[str] = []   # bisher extrahierte Überschriften

            n_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                check_cancel()
                progress(20 + int(i / n_chunks * 40), f"Verarbeite Chunk {i + 1}/{n_chunks}...")

                # ── Writing cache: skip LLM call if result already cached ──────
                c_key = get_text_writing_cache_key(chunk, model_id, detail_level)
                cached_text = load_text_writing_cache(c_key)
                if cached_text is not None:
                    log(f"   💾 Chunk {i + 1} aus Cache geladen.")
                    cleaned = cached_text
                    processed_texts.append(cleaned)
                    import re as _re
                    for m in _re.finditer(r'#{2,3}\s+(.+)', cleaned):
                        h = m.group(1).strip()
                        if h and h not in covered_headers:
                            covered_headers.append(h)
                    if preview_callback:
                        inter_md = assemble_final_markdown(
                            parts=processed_texts,
                            title=f"{stem} (In Bearbeitung...)",
                            detail_level=detail_level,
                            source_files=[source_path.name],
                        )
                        preview_callback(inter_md)
                    continue

                # ctx_before: verarbeiteter Vorgänger-Chunk (letzte Zeichen) + Themen-Liste
                if i > 0 and processed_texts:
                    raw_before = extract_last_n_chars(processed_texts[-1], config.CONTEXT_CHARS)
                    if covered_headers:
                        topics_hint = "BEREITS BEHANDELTE THEMEN: " + ", ".join(covered_headers[-8:]) + "\n"
                        ctx_before = topics_hint + f"KONTEXT VORHERIGER ABSCHNITT (Ende):\n{raw_before}"
                    else:
                        ctx_before = raw_before
                else:
                    ctx_before = ""
                ctx_after = extract_first_n_chars(chunks[i + 1] if i + 1 < n_chunks else "", config.CONTEXT_CHARS)

                if executor:
                    executor.reset_section_counters()

                processed = worker_post_process_text(
                    client=client,
                    text_chunk=chunk,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    context_before=ctx_before,
                    context_after=ctx_after,
                    tools=tools,
                    tool_executor=executor,
                )
                cleaned = clean_llm_markdown_output(processed)
                save_text_writing_cache(c_key, cleaned)
                processed_texts.append(cleaned)

                # Überschriften aus diesem Chunk extrahieren und merken
                import re as _re
                for m in _re.finditer(r'#{2,3}\s+(.+)', cleaned):
                    h = m.group(1).strip()
                    if h and h not in covered_headers:
                        covered_headers.append(h)

                # Progressive Update der Vorschau
                if preview_callback:
                    inter_md = assemble_final_markdown(
                        parts=processed_texts,
                        title=f"{stem} (In Bearbeitung...)",
                        detail_level=detail_level,
                        source_files=[source_path.name],
                    )
                    preview_callback(inter_md)
        else:
            processed_texts = chunks

        # ── PHASE B2: LLM-DOKUMENTTITEL ──────────────────────────────────────
        # Generiere einen akademischen Titel aus dem Inhalt statt Dateiname zu nutzen.
        llm_title = stem  # Fallback: Dateiname
        if processed_texts and model_id:
            try:
                title_snippet = (
                    extract_first_n_chars(processed_texts[0], 600)
                    + ("\n...\n" + extract_last_n_chars(processed_texts[-1], 300)
                       if len(processed_texts) > 1 else "")
                )
                title_prompt = (
                    "Erstelle einen kurzen, akademischen Titel für ein medizinisches Lernskript "
                    "mit folgendem Inhalt:\n"
                    f"\"\"\"{title_snippet}\"\"\"\n"
                    "Antworte NUR mit dem Titel (keine Anführungszeichen, kein Punkt am Ende, max. 8 Wörter)."
                )
                raw_title = robust_chat_completion(
                    client, model_id,
                    [{"role": "user", "content": title_prompt}],
                    max_tokens=config.UTILITY_MAX_TOKENS,
                    temperature=0.1,
                ).strip().replace('"', "").replace("'", "")
                if raw_title and len(raw_title) < 120:
                    llm_title = raw_title
                    log(f"   📖 Dokumenttitel: {llm_title}")
            except Exception:
                pass  # Fallback bleibt stem

        # ── PHASE C: VISION PROCESSING ────────────────────────────────────────
        progress(62, "Vision-Modell lädt...")
        vision_entries = [r for r in registry if r["type"] in ("image", "table")]

        if vision_entries:
            log(f"👁️ Verarbeite {len(vision_entries)} Bild(er)/Tabelle(n)...")
            v_mod = settings.get("vision_model", config.VISION_MODEL_LOAD)
            vis_model_id = switch_model(client, v_mod, v_mod)
            img_counter = 0

            for i, entry in enumerate(vision_entries):
                check_cancel()
                progress(62 + int(i / len(vision_entries) * 25), f"Bild/Tabelle {i + 1}/{len(vision_entries)}...")
                pil_img = entry.get("image")
                if pil_img is None:
                    continue

                img_hash_key = get_image_hash(pil_img)
                cached = cache.get(img_hash_key)

                if cached:
                    entry["processed"] = cached
                    continue

                if entry["type"] == "table":
                    result = worker_analyze_table(client, pil_img, vis_model_id)
                else:
                    result = worker_analyze_image(client, pil_img, vis_model_id, entry.get("context", ""), detail_level)

                cache.set(img_hash_key, result)
                entry["processed"] = result

                # Bild speichern
                if entry["type"] == "image":
                    action = result.get("action", "SKIP") if isinstance(result, dict) else "SKIP"
                    if action != "SKIP":
                        img_counter += 1
                        img_path = images_dir / f"img_{img_counter:03d}.jpg"
                        save_pil_image(pil_img, img_path)
                        if isinstance(result, dict):
                            result["_saved_path"] = str(img_path)

        # ── PHASE D: ASSEMBLY ─────────────────────────────────────────────────
        progress(88, "Dokument zusammenstellen...")

        def _entry_to_md(entry: dict) -> str:
            """Konvertiert einen verarbeiteten Bild-/Tabellen-Registry-Eintrag zu Markdown."""
            result = entry.get("processed")
            if result is None:
                return ""
            if entry["type"] == "table":
                if result and not result.startswith("[FEHLER"):
                    return result
                return ""
            elif isinstance(result, dict):
                action = result.get("action", "SKIP")
                if action == "SKIP":
                    return ""
                caption = result.get("caption", "")
                analysis = result.get("analysis", "")
                saved = result.get("_saved_path", "")
                md_parts = []
                if saved:
                    caption_html = markdown.markdown(caption)
                    if caption_html.startswith("<p>") and caption_html.endswith("</p>"):
                        caption_html = caption_html[3:-4]
                    md_parts.append(f"![{caption}](images/{Path(saved).name})")
                    md_parts.append(f"<p class='image-caption'>{caption_html}</p>")
                if action == "KEEP_ANALYSIS" and analysis:
                    analysis_html = markdown.markdown(analysis)
                    md_parts.append(
                        f"\n<div class='analysis-box'>"
                        f"<span class='analysis-title'>🔍 Bild-Analyse:</span> {analysis_html}</div>"
                    )
                return "\n".join(md_parts)
            return ""

        # Inline-Bild-Placement: Bilder/Tabellen nach dem Chunk einordnen,
        # nach dem ihr ursprünglicher Text-Nachbar im Dokument erscheint.
        # image_after_chunk[i] = Liste von Markdown-Strings die nach Chunk i eingefügt werden.
        # image_after_chunk[-1] = vor dem ersten Chunk.
        image_after_chunk: dict[int, list[str]] = {}
        for entry in registry:
            if entry["type"] not in ("image", "table"):
                continue
            img_md = _entry_to_md(entry)
            if not img_md:
                continue
            j = entry.get("text_idx_before", 0)
            if j > 0:
                ci = _te_to_chunk(j - 1)  # nach dem letzten Text-Entry vor diesem Bild
            else:
                ci = -1  # vor allen Chunks
            image_after_chunk.setdefault(ci, []).append(img_md)

        # Assembly in Registry-Reihenfolge: Bilder inline zwischen Text-Chunks
        all_parts: list[str] = []
        for img_md in image_after_chunk.get(-1, []):
            all_parts.append(img_md)
        for i, pt in enumerate(processed_texts):
            all_parts.append(pt)
            for img_md in image_after_chunk.get(i, []):
                all_parts.append(img_md)
        final_md = assemble_final_markdown(
            parts=all_parts,
            title=llm_title,
            detail_level=detail_level,
            source_files=[source_path.name],
        )

        # RAG-Image-Tags auflösen
        final_md = resolve_rag_image_tags(final_md, images_dir)

        # Ausgabe
        progress(93, "PDF / Markdown speichern...")
        output_path = None
        if output_format in ("pdf", "both"):
            pdf_path = output_dir / f"{stem}.pdf"
            ok = markdown_to_pdf(final_md, pdf_path, output_dir, detail_level)
            if ok:
                output_path = pdf_path
                log(f"✅ PDF gespeichert: {pdf_path}")
            else:
                log("⚠️ PDF-Erstellung fehlgeschlagen, speichere Markdown...")
        if output_format in ("md", "both") or output_path is None:
            md_path = output_dir / f"{stem}.md"
            save_markdown(final_md, md_path)
            if output_path is None:
                output_path = md_path
            log(f"✅ Markdown gespeichert: {md_path}")

        # Optional Anki export — creates simple flashcards from bold terms in the finished markdown
        if settings.get("anki_export", False):
            from modes.synthesis import _try_anki_export
            _try_anki_export([final_md], output_dir, llm_title, log)

        # Evidence PDF is not available in single-file summary mode —
        # it is a traceability feature for multi-source synthesis only.
        if settings.get("evidence_pdf", False):
            log("ℹ️ Evidence-PDF: Im Einzeldatei-Modus nicht verfügbar.")

        progress(100, "✅ Fertig!")
        log(f"✅ Verarbeitung abgeschlossen: {source_path.name}")
        return {"success": True, "output_path": output_path, "markdown": final_md, "error": ""}

    except InterruptedError as e:
        if log_callback:
            log_callback(f"⏹ {e}")
        return {"success": False, "output_path": None, "markdown": "", "error": str(e)}
    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        if log_callback:
            log_callback(f"❌❌❌ Kritischer Fehler: {e}")
        return {"success": False, "output_path": None, "markdown": "", "error": error_msg}


# ── Worker-Funktionen ─────────────────────────────────────────────────────────

def worker_post_process_text(
    client,
    text_chunk: str,
    model_id: str,
    system_prompt: str,
    context_before: str = "",
    context_after: str = "",
    tools: list[dict] = None,
    tool_executor: ToolExecutor = None,
) -> str:
    """Sendet Text-Chunk zum LLM zur Strukturierung, Korrektur und Anreicherung."""
    user_msg = (
        f"HINTERGRUNDWISSEN (KONTEXT DAVOR):\n{context_before}\n---\n"
        f"TEXT ZUR BEARBEITUNG:\n{text_chunk}\n---\n"
        f"HINTERGRUNDWISSEN (KONTEXT DANACH):\n{context_after}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    if tools and tool_executor and config.AGENT_TOOLS_ENABLED:
        text, _ = agentic_chat_completion(
            client=client,
            model=model_id,
            messages=messages,
            tools=tools,
            tool_executor=tool_executor.execute,
            temperature=0.1,
        )
        return text
    else:
        return robust_chat_completion(client, model_id, messages, temperature=0.1)


def worker_analyze_image(
    client,
    pil_image,
    model_id: str,
    context: str,
    detail_level: int,
) -> dict:
    """
    Analysiert ein Bild mit dem Vision-Modell.
    Gibt {'action': 'SKIP'|'KEEP_SIMPLE'|'KEEP_ANALYSIS', 'caption': str, 'analysis': str} zurück.
    """
    prompt = get_image_prompt(detail_level, context)
    img_b64 = pil_to_base64_jpeg(pil_image)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ]},
    ]
    raw = robust_chat_completion(client, model_id, messages)
    raw = raw.strip()

    if raw.upper().startswith("SKIP"):
        return {"action": "SKIP", "caption": "", "analysis": ""}

    if raw.upper().startswith("KEEP_SIMPLE"):
        caption = raw.split(":", 1)[1].strip() if ":" in raw else raw[11:].strip()
        return {"action": "KEEP_SIMPLE", "caption": caption, "analysis": ""}

    if raw.upper().startswith("KEEP_ANALYSIS"):
        body = raw.split(":", 1)[1].strip() if ":" in raw else raw[13:].strip()
        # Ersten Satz als Caption
        first_dot = body.find(".")
        caption = body[:first_dot + 1].strip() if first_dot != -1 else body[:80]
        return {"action": "KEEP_ANALYSIS", "caption": caption, "analysis": body}

    # Fallback
    return {"action": "KEEP_SIMPLE", "caption": raw[:200], "analysis": ""}


def worker_analyze_table(
    client,
    pil_image,
    model_id: str,
) -> str:
    """Analysiert eine Tabelle, gibt Markdown-Tabelle zurück."""
    img_b64 = pil_to_base64_jpeg(pil_image)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": TABLE_PROMPT},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ]},
    ]
    return robust_chat_completion(client, model_id, messages)
