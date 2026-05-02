# modes/ocr_synthesis.py — OCR mode: multiple files to synthesized textbook via page images

# OCR mode: Multiple files → synthesized textbook via page images.
# IMPORTANT: run_ocr_synthesis_worker() must run in separate process (spawn).

from __future__ import annotations

import multiprocessing
import traceback
from pathlib import Path

import config


# ── Prompt-Templates ─────────────────────────────────────────────────────────

_MASTER_UNIFICATION_PROMPT = """\\
Du erhältst eine JSON-Übersicht über Vorlesungsfolien aus {n} Dateien.
Jede Datei hat eine Liste von Themengruppen mit main_topic, sub_topic, page_count und is_new_main_topic.
{document_topics_line}
is_new_main_topic=true: Das Analysis-Modell hat hier einen echten Kapitelbruch erkannt.
Nutze dieses Signal als starkes Indiz für Kapitelgrenzen beim Zusammenführen.

Erstelle eine EINHEITLICHE, DEDUPLIZIERTE, HIERARCHISCHE Themenstruktur für ein synthetisches Lernskript.

REGELN für main_topic:
1. KLINISCHE KATEGORIE / KAPITELNAME — repräsentiert eine inhaltliche Einheit (z.B. "Vaskulitiden", "Kollagenosen", "Rheumatoide Arthritis").
2. NIEMALS das übergeordnete Fachgebiet als main_topic wenn alles dazu gehört (z.B. NICHT "Rheumatologie", "Innere Medizin").
3. STRIKT VERBOTEN als main_topic oder Zusatz: "Übersicht", "Einleitung", "Allgemeines", "Grundlagen", "Einführung".
   → Wenn eine Gruppe nur eine Übersichtsfolie enthält: weise sie dem inhaltlich passenden Kapitel zu, nicht einem "Übersicht"-Kapitel.
4. AGGRESSIVE DEDUPLIZIERUNG: Fasse semantisch identische oder stark verwandte Formulierungen zusammen.
   Beispiele: "Systemische Sklerose" + "Systemische Sklerose — Übersicht" + "SSc" → ALLE zu "Systemische Sklerose".
   "RA-Therapie" + "Rheumatoide Arthritis — Therapie" + "Biologika bei RA" → alle zu "Rheumatoide Arthritis".
5. Ziel: 8–12 Hauptthemen gesamt — lieber breiter fassen als zu viele Kleinstkapitel.
6. Logische Reihenfolge: Grundlagen → Pathogenese → Klinik → Diagnostik → Therapie → Spezifische Erkrankungen.

REGELN für sub_topic:
- Spezifischer Aspekt des Hauptthemas: Format "Thema — Aspekt" (z.B. "Kollagenosen — Diagnostik").
- Kurz und prägnant, kein Strukturwort am Ende (NICHT "Kollagenosen — Übersicht").

Gib ein JSON-Array zurück:
[
  {{
    "main_topic": "Einheitliche Kapitelüberschrift (klinische Kategorie)",
    "sub_topic": "Spezifischer Aspekt — ohne Strukturwörter",
    "sources": [
      {{"file_index": 0, "group_indices": [0, 1]}},
      {{"file_index": 1, "group_indices": [2]}}
    ]
  }},
  ...
]

Weitere Regeln:
- "group_indices": Indizes der PageGroups aus file_analyses[file_index].page_groups.
- NIEMALS 'Abschnitt', 'Kapitel' oder Nummern in main_topic oder sub_topic.
- Nur JSON zurückgeben, kein erklärender Text.

Folien-Übersicht:
{summary_json}
"""

def get_ocr_synthesis_prompt(detail_level: int, detail_desc: str, context_block: str, figure_block: str, main_topic: str, sub_topic: str, source_files: str) -> str:
    """Erstellt den System-Prompt für OCR Deep Synthesis mit abgestuftem Detailgrad."""
    base = (
        "Du bist ein erfahrener Hochschuldozent für Medizin und schreibst ein synthetisches Lernscript "
        "aus mehreren Vorlesungsreihen. Deine Aufgabe: Schreibe einen vollständigen, didaktisch "
        "aufbereiteten Abschnitt zum angegebenen Thema, basierend auf den Vorlesungsfolien aus allen Quellen.\n\n"
        "DEINE AUFGABEN:\n"
        "1. SPRACHE:\n"
        "   - Korrigiere eventuelle OCR-Lese-Fehler aus den Vorlesungsfolien selbstständig.\n"
        "   - Schreibe auf Deutsch, akademisch aber verständlich.\n"
        "2. STRUKTUR & LAYOUT:\n"
        "   - FORMATIERUNG: Nutze **Fettungen** für wichtige Fachbegriffe, Medikamente oder Schlüsselkonzepte.\n"
        "   - LISTEN: Nutze Bullet-Points für Aufzählungen oder Schritte, um Bleiwüsten zu vermeiden.\n"
        "   - ABSÄTZE: Mache spätestens alle 5-6 Sätze einen sinnvollen Absatz.\n"
        "   - ÜBERSCHRIFTEN: Nutze Markdown (## für Hauptüberschriften, ### für Unterabschnitte). "
        "Verwende NIEMALS Präfixe wie 'Abschnitt' oder Nummerierungen in den Überschriften. Beginne direkt mit dem Fachbegriff!\n"
        "   - MERKSÄTZE: Wenn etwas besonders wichtig ist, nutze einen Zitat-Block (> Merksatz: ...).\n"
        "3. ABBILDUNGEN EINBINDEN:\n"
        "   - Binde alle sichtbaren Diagramme, Tabellen und Abbildungen der Folien aktiv in deinen Text ein!\n"
    )

    if detail_level >= 90:
        instruction = (
            f"4. DETAILS (INHALT \u2014 100% TREUE / {detail_level}%):\n"
            "   - STIL: Schreibe in vollständigen, akademischen Sätzen. Wandle Stichpunkte in Fließtext um.\n"
            "   - Behalte JEDES fachliche Detail der Folien. Entferne keine Zahlen, Dosen oder Mechanismen.\n"
            "   - Erkläre Mechanismen, Zusammenhänge und klinische Relevanz.\n"
            "   - Integriere komplementäre Informationen aus den verschiedenen Quellen sinnvoll.\n"
            "   - Der Text muss ein Lehrbuch ersetzen können. Maximale Ausführlichkeit bei perfekter Lesbarkeit.\n"
        )
    elif detail_level >= 40:
        instruction = (
            f"4. DETAILS (INHALT \u2014 ZUSAMMENFASSUNG / {detail_level}%):\n"
            "   - STIL: Gut lesbarer Fließtext, strukturiert durch Absätze.\n"
            "   - Behalte ALLE Informationen der Folien, aber fasse sie kompakter zusammen.\n"
            "   - Entferne redundante Füllwörter, aber behalte alle Fakten, Zahlen und Dosen.\n"
            "   - Führe parallele Themen aus verschiedenen Quellen sauber zusammen.\n"
        )
    else:
        instruction = (
            f"4. DETAILS (INHALT \u2014 CHEAT-SHEET / {detail_level}%):\n"
            "   - STIL: Nutze primär Stichpunkte (Bulletpoints). Fließtext nur für kurze Zusammenfassungen.\n"
            "   - REDUZIERE radikal auf das Wesentliche der Folien.\n"
            "   - Behalte NUR harte Fakten, Schlüsselbegriffe, Zahlen und Dosen.\n"
        )

    constraints = "   - MATHE: Formeln müssen in Sprache überführt werden (z.B. Bruch als \"A geteilt durch B\"). Kein LaTeX.\n\n"

    tools_block = (
        "TOOLS ZUR WISSENSANREICHERUNG:\n"
        "- insert_figure: AKTIV NUTZEN um passende Abbildungen aus den Folien einzubinden \u2014 sehr wichtig!\n"
        "  Binde alle relevanten Abbildungen aus der Verfügbarkeitsliste unten in deinen Text ein.\n"
        "- search_knowledge_base: Nutze es, um Pathomechanismen, Hintergründe und Details zu vertiefen.\n"
        "- search_web: Nutze es für aktuelle Leitlinien, Grenzwerte, epidemiologische Daten oder spezifische Fakten.\n"
        "- Wenn ein Tool-Ergebnis nicht relevant ist: ignoriere es, schreibe trotzdem weiter.\n\n"
    )

    data_block = (
        f"{context_block}{figure_block}\n"
        f"AKTUELLES THEMA: **{main_topic}** \u2014 {sub_topic}\n"
        f"Quellen: {source_files}\n\n"
        f"WICHTIG ZUM THEMA: Du schreibst einen vollständigen, eigenständigen Abschnitt "
        f"über '{sub_topic or main_topic}'. Auch wenn es sich um ein Teilthema eines größeren "
        f"Komplexes handelt: Behandle es VOLLSTÄNDIG mit Ätiologie, Pathophysiologie, Klinik, "
        f"Diagnostik und Therapie — soweit die Folien dazu Informationen enthalten.\n"
    )

    return base + instruction + constraints + tools_block + data_block

_WRITING_USER_SYNTHESIS = """\
Schreibe jetzt den Lernscript-Abschnitt über "{main_topic} — {sub_topic}" \
basierend auf diesen {n} Vorlesungsfolien aus {n_files} Quelle(n).
Beginne direkt mit dem Markdown-Inhalt.
"""


# ── Haupt-API ─────────────────────────────────────────────────────────────────

def process_multiple_files_ocr(
    source_paths: list[Path],
    settings: dict,
    progress_callback: callable | None = None,
    log_callback: callable | None = None,
    cancel_event=None,
    preview_callback: callable | None = None,
) -> dict:
    """
    Startet OCR Deep Synthesis in einem Worker-Prozess (spawn).
    Gibt {'success': bool, 'output_path': Path|None, 'markdown': str, 'error': str} zurück.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    log_queue = ctx.Queue()
    preview_queue = ctx.Queue()

    worker = ctx.Process(
        target=run_ocr_synthesis_worker,
        args=(source_paths, settings, result_queue, progress_queue, log_queue, preview_queue),
        daemon=False,
    )
    worker.start()

    while worker.is_alive() or not result_queue.empty():
        if cancel_event and cancel_event.is_set():
            worker.terminate()
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.kill()
            if log_callback:
                log_callback("⚠️ OCR-Synthese durch Benutzer abgebrochen.")
            return {"success": False, "output_path": None, "markdown": "", "error": "Vorgang durch Benutzer abgebrochen."}
            
        while not progress_queue.empty():
            try:
                pct, msg = progress_queue.get_nowait()
                if progress_callback:
                    progress_callback(pct, msg)
            except Exception:
                pass
        while not log_queue.empty():
            try:
                msg = log_queue.get_nowait()
                if log_callback:
                    log_callback(msg)
                else:
                    print(msg)
            except Exception:
                pass
        while not preview_queue.empty():
            try:
                md_text = preview_queue.get_nowait()
                if preview_callback:
                    preview_callback(md_text)
            except Exception:
                pass
        if not result_queue.empty():
            break
        worker.join(timeout=0.2)

    worker.join(timeout=120)

    if not result_queue.empty():
        return result_queue.get()
    return {"success": False, "output_path": None, "markdown": "",
            "error": "Worker-Prozess ohne Ergebnis beendet"}


# ── Worker-Funktion ───────────────────────────────────────────────────────────

def run_ocr_synthesis_worker(
    source_paths: list[Path],
    settings: dict,
    result_queue,
    progress_queue,
    log_queue,
    preview_queue,
) -> None:
    # WICHTIG: Einstellungen neu laden (für Multiprocessing-Spawn)
    from core.settings_manager import get_settings
    get_settings()
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _progress(pct: int, msg: str) -> None:
        progress_queue.put((pct, msg))

    def _log(msg: str) -> None:
        log_queue.put(msg)
        print(msg, flush=True)

    try:
        _run_ocr_synthesis_pipeline(source_paths, settings, _progress, _log, result_queue, preview_queue)
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"❌ OCR-Synthesis-Worker Fehler: {e}\n{tb}")
        result_queue.put({"success": False, "output_path": None, "markdown": "", "error": str(e)})


def _run_ocr_synthesis_pipeline(
    source_paths: list[Path],
    settings: dict,
    progress: callable,
    preview_queue,
) -> None:
    # WICHTIG: In neuem Prozess (Multiprocessing) müssen Einstellungen neu geladen werden,
    # da das config-Modul sonst die Default-Werte aus der Datei hat.
    from core.settings_manager import get_settings
    get_settings()

    import json as _json

    from core.llm_client import (
        create_openai_client, ensure_lm_studio_running,
        switch_model, agentic_chat_completion, robust_chat_completion,
        extract_json_robust,
    )
    from core.pdf_parser import repair_pdf_if_needed
    from core.image_utils import pil_to_base64_jpeg, AdvancedDeduplicator
    from core.output import assemble_final_markdown, save_markdown, markdown_to_pdf
    from core.text_utils import clean_llm_markdown_output
    from core.page_renderer import (
        get_page_count, render_pages, render_all_pages_as_thumbnails,
        extract_figures_from_pdf, get_figures_for_pages, build_figure_list_text,
        load_analysis_cache, save_analysis_cache,
        PageGroup, OCRFileAnalysis, OCRUnifiedTopic,
    )
    from rag.tools import ToolExecutor
    from rag.ocr_tools import OCRToolExecutor
    from rag.augmenter import resolve_rag_image_tags
    # Import ocr_summary helpers (nur was tatsächlich genutzt wird)
    from modes.ocr_summary import (
        _run_analysis_batch, _build_page_groups,
        _make_fallback_groups, _detect_document_topic,
    )

    # ── Einstellungen ───────────────────────────────────────────────────────
    detail_level    = settings.get("detail_level", 80)
    rag_store_name  = settings.get("rag_store_name", None)
    web_enabled     = settings.get("web_search_enabled", False)
    output_format   = settings.get("output_format", "pdf")
    writing_dpi     = settings.get("ocr_writing_dpi", config.OCR_WRITING_DPI)
    max_pages_write = settings.get("ocr_max_pages_per_call", config.OCR_MAX_PAGES_PER_WRITING_CALL)
    project_name    = settings.get("project_name", "Synthese_OCR")
    detail_desc     = "umfassend und ausführlich" if detail_level > 60 else "kompakt und präzise"

    # ═════════════════════════════════════════════════════════════
    # PHASE A — VORBEREITUNG
    # ═════════════════════════════════════════════════════════════
    progress(2, "Prüfe LM Studio...")
    if not ensure_lm_studio_running():
        result_queue.put({"success": False, "output_path": None, "markdown": "",
                          "error": "LM Studio nicht erreichbar"})
        return

    output_dir = Path(config.CENTRAL_OUTPUT_DIR) / f"{project_name}_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = create_openai_client()

    # PDFs reparieren, Seitenanzahlen prüfen
    work_paths: list[Path] = []
    repaired_map: dict[Path, Path] = {}  # work_path → original für Cleanup
    for sp in source_paths:
        r = repair_pdf_if_needed(sp)
        work_paths.append(r)
        if r != sp:
            repaired_map[r] = sp
        pc = get_page_count(r)
        log(f"📄 {sp.name}: {pc} Seiten")

    # ═════════════════════════════════════════════════════════════
    # PHASE B — PRO-DATEI: DOCLING + ANALYSIS PASS
    # ═════════════════════════════════════════════════════════════
    progress(5, "Lade Vision-Modell...")
    v_mod = settings.get("vision_model", config.VISION_MODEL_LOAD)
    vision_model_id = switch_model(client, v_mod, v_mod)

    file_analyses: list[OCRFileAnalysis] = []
    file_document_topics: list[str] = []   # per-file detected subjects (for master prompt)
    n_files = len(work_paths)

    for file_idx, work_path in enumerate(work_paths):
        file_start_pct = 5 + int((file_idx / n_files) * 25)
        orig_name = source_paths[file_idx].name

        progress(file_start_pct, f"Verarbeite Datei {file_idx + 1}/{n_files}: {orig_name}")
        log(f"\n{'=' * 50}")
        log(f"📁 Datei {file_idx + 1}/{n_files}: {orig_name}")

        fig_dir = output_dir / f"figures_{work_path.stem}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Docling Figuren
        log("   🔍 Docling Figuren-Extraktion...")
        deduplicator = AdvancedDeduplicator()
        figures = extract_figures_from_pdf(work_path, fig_dir, deduplicator)

        # Analysis Cache prüfen
        cached = load_analysis_cache(work_path)
        if cached is not None:
            page_groups = cached.page_groups
            log(f"   ✅ Cache-Hit: {len(page_groups)} Gruppen")
            file_document_topics.append("")   # topic unknown for cached files
        else:
            page_count = get_page_count(work_path)
            thumbnails = render_all_pages_as_thumbnails(work_path, dpi=config.OCR_ANALYSIS_DPI)
            if not thumbnails:
                log("   ⚠️ Rendering fehlgeschlagen — Fallback")
                page_groups = _make_fallback_groups(work_path, page_count, config.OCR_FALLBACK_GROUP_SIZE)
                file_document_topics.append("")
            else:
                # Detect overarching lecture subject for this file
                doc_topic = _detect_document_topic(thumbnails, client, vision_model_id)
                if doc_topic:
                    log(f"   🎯 Vorlesungsthema erkannt: {doc_topic}")
                file_document_topics.append(doc_topic)

                all_page_data: list[dict] = []
                batch_size = config.OCR_MAX_PAGES_PER_ANALYSIS_BATCH
                batches = [thumbnails[i:i + batch_size] for i in range(0, len(thumbnails), batch_size)]

                prev_main_topic = ""
                for b_idx, batch in enumerate(batches):
                    start_p = b_idx * batch_size
                    log(f"   🔎 Analysis Batch {b_idx + 1}/{len(batches)}")
                    batch_data = _run_analysis_batch(
                        batch, start_p, client, vision_model_id,
                        prev_context=prev_main_topic,
                        document_topic=doc_topic,
                        batch_nr=b_idx + 1,
                        total_batches=len(batches),
                    )
                    if batch_data:
                        all_page_data.extend(batch_data)
                        prev_main_topic = batch_data[-1].get("main_topic", "")
                    else:
                        fallback_entries = []
                        for i in range(len(batch)):
                            fallback_entries.append({
                                "page": start_p + i,
                                "main_topic": f"Abschnitt {(start_p + i) // config.OCR_FALLBACK_GROUP_SIZE + 1}",
                                "sub_topic": "",
                                "new_main_topic": (i == 0),
                            })
                        all_page_data.extend(fallback_entries)
                        # Kontinuität erhalten auch bei Fallback
                        prev_main_topic = fallback_entries[-1]["main_topic"] if fallback_entries else prev_main_topic
                page_groups = _build_page_groups(all_page_data, work_path, max_pages_write)

            save_analysis_cache(OCRFileAnalysis(
                file_path=work_path,
                page_count=get_page_count(work_path),
                page_groups=page_groups,
            ))

        # Figuren zuordnen
        for g in page_groups:
            g.figures = get_figures_for_pages(figures, g.pages)

        file_analyses.append(OCRFileAnalysis(
            file_path=work_path,
            page_count=get_page_count(work_path),
            page_groups=page_groups,
        ))
        log(f"   ✅ {len(page_groups)} Themengruppen")

    # ═════════════════════════════════════════════════════════════
    # PHASE C — MASTER-LLM: UNIFIED TOPIC STRUCTURE
    # ═════════════════════════════════════════════════════════════
    progress(32, "Master-LLM: Themenstruktur vereinheitlichen...")
    log("\n🧠 Master-LLM: Themenstruktur vereinheitlichen...")

    # Text-Modell laden für Master-Pass (kein Vision nötig)
    t_mod = settings.get("text_model", config.TEXT_MODEL_LOAD)
    text_model_id = switch_model(client, t_mod, t_mod)

    unified_topics = _run_master_unification(
        file_analyses, client, text_model_id, log,
        document_topics=file_document_topics,
    )
    log(f"   ✅ {len(unified_topics)} vereinheitlichte Topics")

    # ── Sub-Architekt: semantische Unterstruktur pro Topic ────────────────────
    # Läuft mit dem bereits geladenen Text-Modell — kein weiterer Modell-Switch.
    # Nur für Topics mit mehreren verschiedenen Sub-Topics sinnvoll.
    progress(34, "Sub-Architekt plant Unterstruktur...")
    sub_arch_count = 0
    for ut in unified_topics:
        if ut.main_topic.lower().strip() in config.OCR_SKIP_TOPICS:
            continue
        sub_plan = _run_sub_architect(ut, file_analyses, client, text_model_id, log)
        if sub_plan:
            ut.sub_sections = sub_plan
            sub_arch_count += 1
    if sub_arch_count:
        log(f"   🏗️ Sub-Architekt: {sub_arch_count} Topics mit Unterstruktur geplant")

    # ═════════════════════════════════════════════════════════════
    # PHASE D — VISION-MODELL FÜR WRITING-PASS
    # ═════════════════════════════════════════════════════════════
    progress(36, "Lade Vision-Modell für Writing-Pass...")
    v_mod = settings.get("vision_model", config.VISION_MODEL_LOAD)
    vision_model_id = switch_model(client, v_mod, v_mod)

    # ═════════════════════════════════════════════════════════════
    # PHASE E — WRITING PASS (pro Unified Topic)
    # ═════════════════════════════════════════════════════════════
    base_executor = ToolExecutor(rag_store_name, log_callback=log, model_id=vision_model_id)
    sections: list[str] = []
    rolling_context = ""
    prev_main_topic = ""
    prev_sub_topic = ""
    seen_main_topics: list[str] = []      # deduplizierte Main-Topic-Liste
    global_inserted_ids: set[str] = set()  # dokumentweite Bildsperre

    def _write_single_batch(
        batch_images: list,
        batch_page_info: list, # [(filename, pageno), ...]
        batch_figures: list,
        batch_source_names: list,
        batch_main_topic: str,
        batch_sub_topic: str,
        batch_rolling_context: str,
        batch_prev_main: str,
        batch_prev_sub: str,
        batch_is_new_topic: bool,
        global_digest_hint: str = "",
        seen_topics_hint: str = "",
    ) -> str:
        """Schreibt einen einzelnen Bild-Batch zu einem Abschnittstext.
        Wird für große Topics aufgerufen um Kontext-Overflow zu vermeiden."""
        nonlocal global_inserted_ids
        from core.page_renderer import get_writing_cache_key, load_writing_cache, save_writing_cache

        # Cache-Check
        c_key = get_writing_cache_key(
            main_topic=batch_main_topic,
            sub_topic=batch_sub_topic,
            page_info=batch_page_info,
            model_id=vision_model_id,
            detail_level=detail_level
        )
        cached_text = load_writing_cache(c_key)
        if cached_text:
            log(f"   ♻️ Cache-Treffer für: {batch_main_topic} — {batch_sub_topic}")
            return cached_text

        _ocr_exec = OCRToolExecutor(
            figures=batch_figures,
            output_figures_dir=output_dir / "figures",
            base_executor=base_executor,
            log_callback=log,
            web_search_enabled=web_enabled,
        )
        _ocr_exec.set_global_inserted_ids(global_inserted_ids)
        base_executor.reset_section_counters()
        _tools = _ocr_exec.get_all_tools()

        _ctx_block = ""
        if batch_rolling_context and batch_prev_main:
            _ctx_block = (
                f"VORHERIGER KONTEXT:\n"
                f"{seen_topics_hint}"
                f"Letztes Thema: {batch_prev_main} — {batch_prev_sub}\n"
                f"Ende letzter Abschnitt: ...{batch_rolling_context[-300:]}\n\n"
            )
        if global_digest_hint:
            _ctx_block += f"BISHER BEHANDELTE THEMEN (Überblick):\n{global_digest_hint}\n\n"
        if batch_is_new_topic:
            _ctx_block += (
                "STRUKTURHINWEIS: Dies ist der Beginn eines NEUEN Hauptthemas. "
                "Starte mit einer kurzen thematischen Einleitung.\n\n"
            )
        elif batch_prev_main:
            _ctx_block += (
                f"STRUKTURHINWEIS: Fortsetzung / Unterthema von '{batch_prev_main}'. "
                "Stelle einen fließenden Anschluss her.\n\n"
            )

        _fig_list_text = build_figure_list_text(batch_figures, _ocr_exec.figure_aliases)
        _fig_block = f"{_fig_list_text}\n\n" if _fig_list_text else ""

        _sys_prompt = get_ocr_synthesis_prompt(
            detail_level=detail_level,
            detail_desc=detail_desc,
            context_block=_ctx_block,
            figure_block=_fig_block,
            main_topic=batch_main_topic,
            sub_topic=batch_sub_topic or "Allgemein",
            source_files=", ".join(batch_source_names),
        )

        _user_content = _build_user_message_content_synthesis(
            page_images=batch_images,
            page_info=batch_page_info,
            main_topic=batch_main_topic,
            sub_topic=batch_sub_topic,
            n_files=len(batch_source_names),
            figures=batch_figures,
            figure_aliases=_ocr_exec.figure_aliases,
        )

        _msgs = [
            {"role": "system", "content": _sys_prompt},
            {"role": "user", "content": _user_content},
        ]

        if _tools and config.AGENT_TOOLS_ENABLED:
            _text, _ = agentic_chat_completion(
                client=client,
                model=vision_model_id,
                messages=_msgs,
                tools=_tools,
                tool_executor=_ocr_exec.execute,
                log_callback=log,
            )
        else:
            _text = robust_chat_completion(client, vision_model_id, _msgs)

        global_inserted_ids.update(_ocr_exec.get_global_inserted_ids())
        final_text = clean_llm_markdown_output(_text)
        save_writing_cache(c_key, final_text)
        return final_text

    total_topics = len(unified_topics)
    writing_start = 38
    writing_end = 88
    sections_since_digest = 0   # Zähler für globalen Kontext-Digest
    global_digest = ""          # Zusammenfassung bisher behandelter Themen

    for topic_idx, topic in enumerate(unified_topics):
        pct = writing_start + int((topic_idx / max(total_topics, 1)) * (writing_end - writing_start))
        progress(pct, f"Schreibe Topic {topic_idx + 1}/{total_topics}: {topic.main_topic}")

        if topic.main_topic.lower().strip() in config.OCR_SKIP_TOPICS:
            log(f"   ⏭️ Übersprungen: {topic.main_topic}")
            continue

        # Alle Seiten aus allen Quellen sammeln
        all_page_images = []
        all_page_info = [] # (filename, pageno)
        source_names = []
        all_topic_figures = list(topic.figures)

        for source in topic.sources:
            fp = source["file_path"]
            pages = source["pages"]  # 0-indexed
            if not pages:
                continue
            imgs = render_pages(fp, pages, dpi=writing_dpi)
            all_page_images.extend(imgs)
            for p in pages:
                all_page_info.append((fp.name, p))
            source_names.append(fp.name)

        if not all_page_images:
            log(f"   ⚠️ Keine Bilder für Topic {topic_idx + 1}, überspringe")
            continue

        log(f"✍️ [{topic_idx + 1}/{total_topics}] {topic.main_topic} — {topic.sub_topic} "
            f"({len(all_page_images)} Seiten, {len(all_topic_figures)} Abbildungen)")

        # ── Sub-Sektion-Planung für mittelgroße / strukturierte Topics ─────────
        # Priorität 1: Sub-Architekt-Ergebnis verwenden (vorab geplant, frac_start/frac_end)
        # Priorität 2: Runtime-Planung für mittelgroße Topics (half_limit < pages <= max_pages)
        max_pages = config.OCR_MAX_PAGES_PER_WRITING_CALL
        half_limit = max_pages // 2
        n_imgs = len(all_page_images)

        def _write_sub_sections(
            sub_items: list[dict],
            title_key: str,
            start_key: str,
            end_key: str,
        ) -> tuple[str, bool]:
            """
            Schreibt eine Liste von Unterabschnitten und gibt (section_text, success) zurück.
            sub_items: Liste mit title_key, start_key (Frac 0-1), end_key (Frac 0-1).
            """
            sub_texts: list[str] = []
            sub_rolling = rolling_context
            sub_prev_m = prev_main_topic
            sub_prev_s = prev_sub_topic
            is_first_sub = True
            for sub_item in sub_items:
                frac_s = float(sub_item.get(start_key, 0.0))
                frac_e = float(sub_item.get(end_key, 1.0))
                page_start = min(int(frac_s * n_imgs), n_imgs)
                page_end = min(int(frac_e * n_imgs), n_imgs)
                if page_start >= page_end:
                    continue
                sub_imgs = all_page_images[page_start:page_end]
                sub_info = all_page_info[page_start:page_end]
                sub_figs = (
                    all_topic_figures[
                        int(frac_s * len(all_topic_figures)):
                        int(frac_e * len(all_topic_figures))
                    ] if all_topic_figures else []
                )
                sub_title = sub_item.get(title_key, topic.sub_topic or "Allgemein")
                if sub_imgs:
                    log(f"   📑 Unterabschnitt: {sub_title} ({len(sub_imgs)} Seiten)")
                    s_text = _write_single_batch(
                        batch_images=sub_imgs,
                        batch_page_info=sub_info,
                        batch_figures=sub_figs,
                        batch_source_names=source_names,
                        batch_main_topic=topic.main_topic,
                        batch_sub_topic=sub_title,
                        batch_rolling_context=sub_rolling,
                        batch_prev_main=sub_prev_m,
                        batch_prev_sub=sub_prev_s,
                        batch_is_new_topic=(is_first_sub and topic.main_topic != prev_main_topic),
                        global_digest_hint=global_digest,
                        seen_topics_hint=("Bereits abgehandelt: " + ", ".join(seen_main_topics[-4:]) + "\n") if seen_main_topics else "",
                    )
                    if s_text and not s_text.startswith("[FEHLER"):
                        sub_texts.append(s_text)
                        sub_rolling = s_text[-config.OCR_ROLLING_CONTEXT_CHARS:]
                        sub_prev_m = topic.main_topic
                        sub_prev_s = sub_title
                is_first_sub = False
            combined = "\n\n".join(sub_texts) if sub_texts else ""
            return combined, bool(combined and not combined.startswith("[FEHLER"))

        # Priorität 1: Sub-Architekt-Ergebnis
        if topic.sub_sections is not None and len(topic.sub_sections) >= 2:
            log(f"   🏗️ Sub-Architekt-Gliederung ({len(topic.sub_sections)} Unterabschnitte)...")
            section_text, ok = _write_sub_sections(
                topic.sub_sections, "sub_title", "frac_start", "frac_end"
            )
            if ok:
                sections.append(section_text)
                rolling_context = section_text[-config.OCR_ROLLING_CONTEXT_CHARS:]
                prev_main_topic = topic.main_topic
                prev_sub_topic = topic.sub_topic
                sections_since_digest += 1
                temp_md = assemble_final_markdown(
                    parts=sections,
                    title=f"{project_name} (In Bearbeitung...)",
                    detail_level=detail_level,
                    source_files=[sp.name for sp in source_paths],
                )
                preview_queue.put(temp_md)
            else:
                log(f"   ⚠️ Sub-Architekt-Abschnitt leer oder Fehler")
            continue  # nächstes Topic

        # Priorität 2: Runtime-Planung für mittelgroße Topics
        if half_limit < n_imgs <= max_pages and len(source_names) > 1:
            log(f"   📋 Runtime-Sub-Sektion-Planung ({n_imgs} Seiten)...")
            sub_plan_prompt = (
                f"Plane die Feinstruktur für den Abschnitt '{topic.main_topic} — {topic.sub_topic or 'Allgemein'}'.\n"
                f"Quellen: {', '.join(source_names)}\n"
                f"Gesamtumfang: {n_imgs} Vorlesungsseiten aus {len(source_names)} Dateien.\n\n"
                "Teile dieses Thema in 2-3 logische Unterabschnitte auf.\n"
                "WICHTIG: Titel OHNE Nummerierung.\n"
                "Antworte NUR als JSON-Array. Jedes Element hat 'title', 'frac_start' und 'frac_end' (0.0–1.0):\n"
                "[{\"title\": \"Anatomie & Grundlagen\", \"frac_start\": 0.0, \"frac_end\": 0.4}, "
                "{\"title\": \"Pathophysiologie\", \"frac_start\": 0.4, \"frac_end\": 1.0}]"
            )
            try:
                sub_plan_resp = robust_chat_completion(
                    client, text_model_id,
                    [{"role": "user", "content": sub_plan_prompt}],
                    max_tokens=config.ANALYSIS_MAX_TOKENS,
                    temperature=0.1,
                )
                sub_plan = extract_json_robust(sub_plan_resp)
            except Exception:
                sub_plan = None

            if sub_plan and isinstance(sub_plan, list) and len(sub_plan) >= 2:
                section_text, ok = _write_sub_sections(
                    sub_plan, "title", "frac_start", "frac_end"
                )
                if ok:
                    sections.append(section_text)
                    rolling_context = section_text[-config.OCR_ROLLING_CONTEXT_CHARS:]
                    prev_main_topic = topic.main_topic
                    prev_sub_topic = topic.sub_topic
                    sections_since_digest += 1
                    temp_md = assemble_final_markdown(
                        parts=sections,
                        title=f"{project_name} (In Bearbeitung...)",
                        detail_level=detail_level,
                        source_files=[sp.name for sp in source_paths],
                    )
                    preview_queue.put(temp_md)
                else:
                    log(f"   ⚠️ Runtime-Sub-Sektion-Abschnitt leer oder Fehler")
                continue  # nächstes Topic

        # ── Seiten-Obergrenze: große Topics in Batches aufteilen ─────────────
        # Verhindert Kontext-Overflow bei Multi-File-Aggregation.
        if len(all_page_images) > max_pages:
            log(f"   ⚡ Topic überschreitet Seiten-Limit ({len(all_page_images)} > {max_pages}) — teile in Batches")
            # Bilder und Figuren proportional auf Batches verteilen
            batch_texts: list[str] = []
            batch_rolling = rolling_context
            batch_prev_m = prev_main_topic
            batch_prev_s = prev_sub_topic
            is_first_batch = True
            for b_start in range(0, len(all_page_images), max_pages):
                b_end = min(b_start + max_pages, len(all_page_images))
                b_imgs = all_page_images[b_start:b_end]
                b_info = all_page_info[b_start:b_end]
                # Figuren proportional zuordnen (grob, da keine Seiten-Zuordnung für Figuren)
                fig_start = int(b_start / len(all_page_images) * len(all_topic_figures))
                fig_end = int(b_end / len(all_page_images) * len(all_topic_figures))
                b_figs = all_topic_figures[fig_start:fig_end]
                b_sub = (f"{topic.sub_topic} (Teil {b_start // max_pages + 1})"
                         if not is_first_batch else topic.sub_topic)
                log(f"   📦 Batch {b_start // max_pages + 1}: Seiten {b_start + 1}–{b_end}")
                b_text = _write_single_batch(
                    batch_images=b_imgs,
                    batch_page_info=b_info,
                    batch_figures=b_figs,
                    batch_source_names=source_names,
                    batch_main_topic=topic.main_topic,
                    batch_sub_topic=b_sub or "Allgemein",
                    batch_rolling_context=batch_rolling,
                    batch_prev_main=batch_prev_m,
                    batch_prev_sub=batch_prev_s,
                    batch_is_new_topic=(is_first_batch and topic.main_topic != prev_main_topic),
                    global_digest_hint=global_digest,
                    seen_topics_hint=("Bereits abgehandelt: " + ", ".join(seen_main_topics[-4:]) + "\n") if seen_main_topics else "",
                )
                if b_text and not b_text.startswith("[FEHLER"):
                    batch_texts.append(b_text)
                    batch_rolling = b_text[-config.OCR_ROLLING_CONTEXT_CHARS:]
                    batch_prev_m = topic.main_topic
                    batch_prev_s = topic.sub_topic or ""
                is_first_batch = False

            section_text = "\n\n".join(batch_texts) if batch_texts else ""
        else:
            # Normaler Einzelaufruf
            section_text = _write_single_batch(
                batch_images=all_page_images,
                batch_page_info=all_page_info,
                batch_figures=all_topic_figures,
                batch_source_names=source_names,
                batch_main_topic=topic.main_topic,
                batch_sub_topic=topic.sub_topic or "Allgemein",
                batch_rolling_context=rolling_context,
                batch_prev_main=prev_main_topic,
                batch_prev_sub=prev_sub_topic,
                batch_is_new_topic=(topic.main_topic != prev_main_topic),
                global_digest_hint=global_digest,
                seen_topics_hint=("Bereits abgehandelt: " + ", ".join(seen_main_topics[-4:]) + "\n") if seen_main_topics else "",
            )

        if section_text and not section_text.startswith("[FEHLER"):
            sections.append(section_text)
            rolling_context = section_text[-config.OCR_ROLLING_CONTEXT_CHARS:]
            prev_main_topic = topic.main_topic
            prev_sub_topic = topic.sub_topic
            if topic.main_topic and topic.main_topic not in seen_main_topics:
                seen_main_topics.append(topic.main_topic)
            sections_since_digest += 1

            # Kontext-Digest: alle OCR_DIGEST_INTERVAL Abschnitte eine komprimierte
            # Übersicht der bisher behandelten Themen erzeugen (gleitendes Fenster).
            if sections_since_digest >= config.OCR_DIGEST_INTERVAL:
                sections_since_digest = 0
                try:
                    window_start = max(0, topic_idx + 1 - config.OCR_DIGEST_INTERVAL * 2)
                    digest_input = "\n".join(
                        f"- {t.main_topic}" + (f" / {t.sub_topic}" if t.sub_topic else "")
                        for t in unified_topics[window_start:topic_idx + 1]
                        if t.main_topic.lower().strip() not in config.OCR_SKIP_TOPICS
                    )
                    digest_prompt = (
                        "Hier sind die bisher behandelten Themen eines medizinischen Lernskripts:\n"
                        f"{digest_input}\n\n"
                        "Schreibe eine 3-5-Satz-Zusammenfassung auf Deutsch: Was wurde bisher behandelt? "
                        "Welche Grundlagen sind damit gelegt? Antworte nur mit dem Fließtext, keine Aufzählung."
                    )
                    global_digest = robust_chat_completion(
                        client, vision_model_id,
                        [{"role": "user", "content": digest_prompt}],
                        max_tokens=config.UTILITY_MAX_TOKENS,
                        temperature=0.1,
                    ).strip()
                    log(f"   📋 Kontext-Digest aktualisiert ({topic_idx + 1} Topics)")
                except Exception:
                    pass  # Digest-Fehler ist unkritisch

            # --- Progressive Preview ---
            temp_md = assemble_final_markdown(
                parts=sections,
                title=f"{project_name} (In Bearbeitung...)",
                detail_level=detail_level,
                source_files=[sp.name for sp in source_paths],
            )
            preview_queue.put(temp_md)
        else:
            log(f"   ⚠️ Abschnitt leer oder Fehler: {section_text[:80]}")

    # ═════════════════════════════════════════════════════════════
    # PHASE F — ASSEMBLY
    # ═════════════════════════════════════════════════════════════
    progress(90, "Erstelle finales Dokument...")

    if not sections:
        result_queue.put({"success": False, "output_path": None, "markdown": "",
                          "error": "Kein Inhalt erzeugt"})
        return

    # Buchtitel via LLM generieren (aus den unified Topic-Namen)
    import re as _re
    all_topic_names = list(dict.fromkeys(
        t.main_topic for t in unified_topics if t.main_topic
    ))[:8]
    topics_str = "; ".join(all_topic_names)
    try:
        title_prompt = (
            f"Erstelle einen kurzen akademischen Buchtitel für ein Lehrbuch mit diesen Themen:\n"
            f"{topics_str}\n"
            "Antworte NUR mit dem Titel (keine Anführungszeichen, kein Punkt am Ende)."
        )
        final_book_title = robust_chat_completion(
            client, text_model_id,
            [{"role": "user", "content": title_prompt}],
            max_tokens=config.UTILITY_MAX_TOKENS,
            temperature=0.1,
        ).strip().replace('"', "").replace("'", "")
        if not final_book_title:
            final_book_title = project_name
    except Exception:
        final_book_title = project_name
    log(f"   📖 Buchtitel: {final_book_title}")

    # Sicherer Dateiname — Umlaut-Transkription vor Sonderzeichen-Entfernung
    _st = final_book_title
    for _src, _dst in [("ä","ae"),("ö","oe"),("ü","ue"),("Ä","ae"),("Ö","oe"),("Ü","ue"),("ß","ss")]:
        _st = _st.replace(_src, _dst)
    safe_title = _re.sub(r"[^\w\s\-]", "", _st).strip()
    safe_title = _re.sub(r"\s+", "_", safe_title)[:80] or project_name

    final_markdown = assemble_final_markdown(
        parts=sections,
        title=final_book_title,
        detail_level=detail_level,
        source_files=[sp.name for sp in source_paths],
    )
    
    # RAG-Image-Tags auflösen (falls Bilder via RAG eingebunden wurden)
    final_markdown = resolve_rag_image_tags(final_markdown, output_dir / "figures")

    md_path = output_dir / f"{safe_title}.md"
    save_markdown(final_markdown, md_path)

    pdf_path = output_dir / f"{safe_title}.pdf"
    if output_format in ("pdf", "both"):
        progress(94, "Erzeuge PDF...")
        markdown_to_pdf(final_markdown, pdf_path, base_url=output_dir, detail_level=detail_level)

    # Temporäre Repair-PDFs löschen
    for rp, orig in repaired_map.items():
        if rp.exists() and rp != orig:
            try:
                rp.unlink()
            except Exception:
                pass

    # Optionaler Evidence-PDF-Export (Traceability)
    if settings.get("evidence_pdf", False):
        log("🔎 Erstelle Evidence-PDF (Quellennachweis)...")
        evidence_md_parts = [
            f"# Quellen-Nachweis zu: {final_book_title}",
            "\n> Dieser Report zeigt die in der OCR-Synthese verarbeiteten Themen-Blöcke und Seiten.\n"
            "> Da die Verarbeitung visuell (als Bild) stattfand, liegt kein nativer Roh-Text vor.\n"
        ]
        for c in file_analyses:
            evidence_md_parts.append(f"## Datei: {c.file_path.name}")
            for blk in c.page_groups:
                pages_str = ", ".join(str(p+1) for p in blk.pages)
                evidence_md_parts.append(f"### {blk.main_topic} - {blk.sub_topic}")
                evidence_md_parts.append(f"**Verarbeitete Seiten:** {pages_str}\n")
                evidence_md_parts.append(f"*[Die Inhalte wurden als Bild-Kontext an das Vision-Modell übergeben.]*\n")

        ev_pdf_path = output_dir / f"{safe_title}_Quellen.pdf"
        ok_ev = markdown_to_pdf(
            "\n".join(evidence_md_parts), ev_pdf_path, output_dir, detail_level
        )
        if ok_ev:
            log(f"✅ Evidence-PDF: {ev_pdf_path}")
        else:
            ev_md_path = output_dir / f"{safe_title}_Quellen.md"
            save_markdown("\n".join(evidence_md_parts), ev_md_path)

    # Optionaler Anki-Export
    if settings.get("anki_export", False):
        try:
            from modes.synthesis import _try_anki_export
            _try_anki_export(sections, output_dir, final_book_title, log)
        except Exception as e:
            log(f"⚠️ Anki-Export Fehler: {e}")

    output_path = pdf_path if (output_format in ("pdf", "both") and pdf_path.exists()) else md_path
    progress(100, "Fertig!")
    log(f"✅ OCR-Synthese abgeschlossen: {final_book_title} → {output_path}")

    result_queue.put({
        "success": True,
        "output_path": output_path,
        "markdown": final_markdown,
        "error": "",
    })


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _run_master_unification(
    file_analyses: list["OCRFileAnalysis"],
    client,
    text_model_id: str,
    log: callable,
    document_topics: list[str] | None = None,
) -> list["OCRUnifiedTopic"]:
    """
    Master-LLM: Empfängt JSON-Zusammenfassung aller PageGroups, gibt unified Topics zurück.
    Fallback: alle Groups der ersten Datei als sequentielle Liste.

    document_topics – optional per-file overarching subjects detected by
                      _detect_document_topic(); used to orient the master LLM.
    """
    import json as _json
    from core.llm_client import robust_chat_completion, extract_json_robust
    from core.page_renderer import OCRUnifiedTopic

    # Zusammenfassung bauen (kein Bild, nur Metadaten)
    summary = {"files": []}
    for file_idx, fa in enumerate(file_analyses):
        groups_summary = []
        for g_idx, g in enumerate(fa.page_groups):
            # Kompaktes Format: kein pages-Array (spart ~80% JSON-Größe bei vielen Gruppen),
            # kein sub_topic_variants (reduziert Rauschen).
            # is_new_main_topic gibt dem Master-LLM das Analysis-Pass-Signal für Kapitelgrenzen.
            groups_summary.append({
                "group_index": g_idx,
                "main_topic": g.main_topic,
                "sub_topic": g.sub_topic,
                "is_new_main_topic": g.is_new_main_topic,
                "page_count": len(g.pages),
            })
        summary["files"].append({
            "file_index": file_idx,
            "filename": fa.file_path.name,
            "page_groups": groups_summary,
        })

    summary_json = _json.dumps(summary, ensure_ascii=False, indent=2)
    # WICHTIG: summary_json enthält { } Zeichen → vor .format() escapen
    # damit Python's str.format() keine KeyError-Ausnahmen wirft
    safe_summary_json = summary_json.replace("{", "{{").replace("}", "}}")

    # Build per-file document-topic context for the master prompt
    known_topics = [
        f"Datei {i+1} ({file_analyses[i].file_path.name}): {t}"
        for i, t in enumerate(document_topics or [])
        if t
    ]
    if known_topics:
        document_topics_line = (
            "\nErkannte Fachgebiete pro Datei (KEINES davon als main_topic verwenden — "
            "gehe eine Ebene tiefer):\n"
            + "\n".join(f"  - {line}" for line in known_topics)
        )
    else:
        document_topics_line = ""

    prompt = _MASTER_UNIFICATION_PROMPT.format(
        n=len(file_analyses),
        summary_json=safe_summary_json,
        document_topics_line=document_topics_line,
    )

    response = robust_chat_completion(
        client=client,
        model=text_model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.ANALYSIS_MAX_TOKENS,
        temperature=0.1,
    )

    parsed = extract_json_robust(response)
    if not isinstance(parsed, list):
        log("   ⚠️ Master-LLM JSON ungültig — Fallback: erste Datei sequentiell")
        return _fallback_unified_topics(file_analyses)

    # OCRUnifiedTopic-Objekte bauen
    unified: list[OCRUnifiedTopic] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = OCRUnifiedTopic(
            main_topic=str(item.get("main_topic", "Unbekannt"))[:80],
            sub_topic=str(item.get("sub_topic", ""))[:80],
            sources=[],
            figures=[],
        )

        for src in item.get("sources", []):
            file_idx = src.get("file_index", -1)
            group_indices = src.get("group_indices", [])
            if file_idx < 0 or file_idx >= len(file_analyses):
                continue
            fa = file_analyses[file_idx]
            pages: list[int] = []
            figs = []
            for g_idx in sorted(group_indices):  # sortiert → korrekte Seitenreihenfolge
                if g_idx < len(fa.page_groups):
                    g = fa.page_groups[g_idx]
                    pages.extend(g.pages)
                    figs.extend(g.figures)
            if pages:
                topic.sources.append({"file_path": fa.file_path, "pages": pages})
                topic.figures.extend(figs)

        if topic.sources:
            unified.append(topic)

    return unified if unified else _fallback_unified_topics(file_analyses)


def _fallback_unified_topics(file_analyses: list) -> list:
    """Fallback wenn Master-LLM fehlschlägt: alle Groups aller Dateien sequentiell."""
    from core.page_renderer import OCRUnifiedTopic
    topics = []
    for fa in file_analyses:
        for g in fa.page_groups:
            topics.append(OCRUnifiedTopic(
                main_topic=g.main_topic,
                sub_topic=g.sub_topic,
                sources=[{"file_path": fa.file_path, "pages": g.pages}],
                figures=g.figures,
            ))
    return topics


def _run_sub_architect(
    topic: "OCRUnifiedTopic",
    file_analyses: list,
    client,
    text_model_id: str,
    log: callable,
) -> list[dict] | None:
    """
    Sub-Architekt: Plant die semantische Unterstruktur für ein einzelnes unified Topic.
    Gibt eine Liste von Sub-Sections zurück oder None bei Fehler/unnötiger Komplexität.

    Jede Sub-Section: {"sub_title": str, "page_fractions": [float, ...]}
    page_fractions: Anteil der Seiten pro Source (Summe ~ 1.0 pro Source).

    Wird nur aufgerufen wenn das Topic Sub-Topics aus mehreren PageGroups hat.
    """
    from core.llm_client import robust_chat_completion, extract_json_robust

    # Sub-Topics aus den zugehörigen PageGroups sammeln
    sub_topic_inputs: list[str] = []
    for src in topic.sources:
        file_idx = next(
            (i for i, fa in enumerate(file_analyses) if fa.file_path == src["file_path"]), -1
        )
        if file_idx < 0:
            continue
        fa = file_analyses[file_idx]
        src_pages = set(src["pages"])
        for g in fa.page_groups:
            if set(g.pages) & src_pages:  # Schnittmenge: diese PageGroup gehört zu diesem Topic
                label = g.main_topic
                if g.sub_topic:
                    label += f" / {g.sub_topic}"
                if label not in sub_topic_inputs:
                    sub_topic_inputs.append(label)

    # Sub-Architekt nur sinnvoll wenn mehrere verschiedene Sub-Topics vorhanden
    if len(sub_topic_inputs) < 2:
        return None

    prompt = (
        f"Plane die Feinstruktur für den Abschnitt '{topic.main_topic}'.\n"
        f"Folgende Quell-Themen sollen integriert werden:\n"
        + "\n".join(f"- {s}" for s in sub_topic_inputs)
        + "\n\nGruppiere diese in 2-4 logische Unterabschnitte mit prägnanten Titeln.\n"
        "WICHTIG: Keine Nummerierung, keine Präfixe wie 'Abschnitt'.\n"
        "Antworte NUR als JSON-Array:\n"
        '[{"sub_title": "Anatomie & Grundlagen"}, {"sub_title": "Pathophysiologie"}]'
    )

    try:
        resp = robust_chat_completion(
            client, text_model_id,
            [{"role": "user", "content": prompt}],
            max_tokens=config.ANALYSIS_MAX_TOKENS,
            temperature=0.1,
        )
        parsed = extract_json_robust(resp)
        if not isinstance(parsed, list) or len(parsed) < 2:
            return None
        result = []
        n = len(parsed)
        for i, item in enumerate(parsed):
            title = str(item.get("sub_title", f"Abschnitt {i+1}"))[:80]
            # Seitenanteil gleichmäßig verteilen (proportional wird im Writing-Pass angewendet)
            frac_start = i / n
            frac_end = (i + 1) / n
            result.append({"sub_title": title, "frac_start": frac_start, "frac_end": frac_end})
        return result
    except Exception:
        return None


def _build_user_message_content_synthesis(
    page_images: list,
    page_info: list, # [(filename, pageno), ...]
    main_topic: str,
    sub_topic: str,
    n_files: int,
    figures: list["DoclingFigure"] = None,
    figure_aliases: dict[str, str] = None,
) -> list[dict]:
    """Baut multimodalen User-Message-Content für den Synthesis Writing-Pass."""
    from core.image_utils import pil_to_base64_jpeg
    from PIL import Image

    content: list[dict] = []
    
    # 1. Vorlesungsfolien (Hauptbilder)
    for i, img in enumerate(page_images):
        info = page_info[i] if i < len(page_info) else ("?", "?")
        fname, pno = info
        b64 = pil_to_base64_jpeg(img)
        content.append({
            "type": "text",
            "text": f"--- Folie {pno} aus Quelle '{fname}' ---",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": b64, "detail": "high"},
        })
        
    # 2. Extrahierte Abbildungen als Orientierungshilfe (Idee A)
    if figures and figure_aliases:
        content.append({
            "type": "text",
            "text": "\n--- ZUR ORIENTIERUNG: VERFÜGBARE ABBILDUNGEN IM DETAIL ---",
        })
        for fig in figures:
            if fig.image_path and fig.image_path.exists():
                try:
                    with Image.open(fig.image_path) as pimg:
                        b64_fig = pil_to_base64_jpeg(pimg.convert("RGB"), quality=70)
                        alias = figure_aliases.get(fig.figure_id, fig.figure_id)
                        
                        content.append({
                            "type": "text",
                            "text": f"Nutze das Tool insert_figure mit ID '{alias}' für dieses Bild:",
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": b64_fig, "detail": "low"},
                        })
                except Exception:
                    pass

    # 3. Text Message
    content.append({
        "type": "text",
        "text": _WRITING_USER_SYNTHESIS.format(
            main_topic=main_topic,
            sub_topic=sub_topic or "Allgemein",
            n=len(page_images),
            n_files=n_files,
        ),
    })
    return content
