# modes/ocr_summary.py — OCR mode: single file to script via page images

# OCR mode: Single file → script via page images (vision model).
# IMPORTANT: run_ocr_summary_worker() must run in separate process (spawn).

from __future__ import annotations

import gc
import multiprocessing
import re
import traceback
from pathlib import Path

import config


# ── Prompt-Templates ─────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM_PROMPT = (
    "Du bist ein Experte für medizinische Hochschullehre. "
    "Analysiere Vorlesungsfolien und erkenne Themen präzise."
)

_ANALYSIS_PROMPT_TEMPLATE = """\\
DEINE ROLLE: Du bist der Gliederungsagent für ein medizinisches Lehrbuch{topic_line}.
{batch_line}
Deine Themen-Einteilungen werden später mit allen anderen Batches zu einem einheitlichen
Inhaltsverzeichnis zusammengeführt. Deine Aufgabe: Finde die logischste Kapitelstruktur
für DIESEN Abschnitt — nicht für die gesamte Vorlesung.
{topic_forbidden_line}
{prev_context}
Analysiere diese {n} Vorlesungsfolien (Seiten {start} bis {end}).
Gib ein JSON-Array zurück — für jede Folie ein Objekt:

[
  {{
    "page": {start},
    "main_topic": "Kollagenosen",
    "sub_topic": "Systemischer Lupus erythematodes — Diagnostik",
    "new_main_topic": true
  }},
  ...
]

REGELN für main_topic:
- Nutze KLINISCHE KATEGORIEN oder KAPITELÜBERSCHRIFTEN — max. 3 Wörter.
- NIEMALS das allgemeine Fachgebiet als main_topic (z.B. "Rheumatologie", "Innere Medizin") — gehe eine Ebene tiefer!
- NIEMALS Struktur-Wörter wie "Übersicht", "Einleitung", "Allgemeines", "Grundlagen" als eigenständigen main_topic.
  Wenn eine Folie eine Übersicht zu einem Thema zeigt, nutze das Thema selbst (z.B. "Kollagenosen", nicht "Kollagenosen — Übersicht").
- Beispiele RICHTIG: "Kollagenosen", "Vaskulitiden", "Pharmakotherapie", "Rheumatoide Arthritis", "Autoimmunologie".
- Beispiele FALSCH: "Rheumatologie" (zu grob), "Psoriasis-Arthritis" (zu spezifisch für Kapitel), "Übersicht" (Strukturwort).
- Gleiche main_topic-Formulierung EXAKT für alle Folien desselben Kapitels — Konsistenz ist essenziell.
- NIEMALS Nummern, Präfixe ("Abschnitt", "Kapitel") in main_topic oder sub_topic.

REGELN für sub_topic:
- Spezifisches Krankheitsbild, Konzept oder Aspekt dieser Folie.
- Format: "Thema — Aspekt", z.B. "Psoriasis-Arthritis — Klinik", "RA — Diagnostik", "Biologika — Wirkmechanismus".
- Nutze IMMER einen Gedankenstrich (—) um Thema und Aspekt zu trennen, wenn beides vorhanden.
- Leer ("") wenn wirklich kein spezifischer Inhalt erkennbar.

SONDERREGEL für Gliederungs- und Titelfolien:
- Titelseiten: main_topic = "Titelseite", sub_topic = "".
- Reine Gliederungsfolien: main_topic = vorheriges Hauptthema beibehalten (Gliederung gehört zum Kapitel!), sub_topic = "Gliederung".

new_main_topic:
- true NUR wenn hier ein NEUES Kapitel beginnt (echter Themenwechsel der Kategorie).
- false bei Folien desselben Themas, auch wenn Aspektwechsel.

Nur JSON zurückgeben, kein erklärender Text davor oder danach.
"""


def _detect_document_topic(
    thumbnails: list,
    client,
    model_id: str,
) -> str:
    """
    Detects the overarching subject of a lecture by sampling a few thumbnails.

    Sends up to 5 evenly-spaced thumbnails to the vision LLM with a minimal
    prompt. Returns a short string like "Rheumatologie" or
    "Kardiologie — Herzinsuffizienz". Falls back to "" on any error so callers
    can treat it as optional context.
    """
    from core.llm_client import robust_chat_completion
    from core.image_utils import pil_to_base64_jpeg

    if not thumbnails:
        return ""

    # Always include slide 0 (often carries the lecture title / agenda directly).
    # Then sample from 4 percentile positions — skipping the very end.
    # Percentile targets: ~5%, ~35%, ~55%, ~85%
    n = len(thumbnails)
    raw_indices = [
        0,                          # title / agenda slide
        max(0, int(n * 0.05)),      # early content
        int(n * 0.35),              # first-third content
        int(n * 0.55),              # mid content
        min(n - 1, int(n * 0.85)), # late content (not the very end)
    ]
    indices = sorted(set(raw_indices))

    content: list[dict] = [{
        "type": "text",
        "text": (
            "Das sind Vorlesungsfolien aus einer medizinischen Lehrveranstaltung.\n"
            "Was ist das übergeordnete Fachgebiet / Thema dieser Vorlesung?\n"
            "Antworte NUR mit 2–4 Wörtern auf Deutsch "
            "(Beispiele: 'Rheumatologie', 'Kardiologie — Herzinsuffizienz', 'Nephrologie').\n"
            "Kein vollständiger Satz, keine Erklärung — nur das Thema."
        ),
    }]

    for idx in indices:
        b64 = pil_to_base64_jpeg(thumbnails[idx], quality=55)
        content.append({
            "type": "image_url",
            "image_url": {"url": b64, "detail": "low"},
        })

    try:
        raw = robust_chat_completion(
            client=client,
            model=model_id,
            messages=[{"role": "user", "content": content}],
            temperature=config.WRITING_TEMPERATURE,
        )
        topic = raw.strip().strip('"').strip("'").strip()
        # Sanity-check: reject if suspiciously long or empty
        if not topic or len(topic) > 80:
            return ""
        return topic
    except Exception:
        return ""

def get_ocr_writing_prompt(detail_level: int, detail_desc: str, context_block: str, figure_block: str, main_topic: str, sub_topic: str) -> str:
    """Erstellt den System-Prompt für den OCR-Schreibdurchgang mit abgestuftem Detailgrad."""
    base = (
        "Du bist ein erfahrener Hochschuldozent für Medizin und schreibst ein hochwertiges Lernscript "
        "für Medizinstudenten. Deine Aufgabe: Schreibe einen vollständigen, didaktisch aufbereiteten "
        "Abschnitt zu dem angegebenen Thema, basierend auf den bereitgestellten Vorlesungsfolien.\n\n"
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
            "   - Der Text muss ein Lehrbuch ersetzen können. Maximale Ausführlichkeit bei perfekter Lesbarkeit.\n"
        )
    elif detail_level >= 40:
        instruction = (
            f"4. DETAILS (INHALT \u2014 ZUSAMMENFASSUNG / {detail_level}%):\n"
            "   - STIL: Gut lesbarer Fließtext, strukturiert durch Absätze.\n"
            "   - Behalte ALLE Informationen der Folien, aber fasse sie kompakter zusammen.\n"
            "   - Entferne redundante Füllwörter, aber behalte alle Fakten, Zahlen und Dosen.\n"
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
    )

    return base + instruction + constraints + tools_block + data_block

_WRITING_USER_TEMPLATE = """\
Schreibe jetzt den Lernscript-Abschnitt über "{main_topic} — {sub_topic}" \
basierend auf diesen {n} Vorlesungsfolien.
Beginne direkt mit dem Markdown-Inhalt (kein Einleitungssatz, keine Erklärung was du tust).
"""




# ── Haupt-API ─────────────────────────────────────────────────────────────────

def process_single_file_ocr(
    source_path: Path,
    settings: dict,
    progress_callback: callable | None = None,
    log_callback: callable | None = None,
    cancel_event=None,
    preview_callback: callable | None = None,
) -> dict:
    """
    Startet OCR-Einzel-Datei-Verarbeitung in einem Worker-Prozess (spawn).
    Gibt {'success': bool, 'output_path': Path|None, 'markdown': str, 'error': str} zurück.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    log_queue = ctx.Queue()
    preview_queue = ctx.Queue()

    worker = ctx.Process(
        target=run_ocr_summary_worker,
        args=(source_path, settings, result_queue, progress_queue, log_queue, preview_queue),
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
                log_callback("⚠️ OCR-Verarbeitung durch Benutzer abgebrochen.")
            return {"success": False, "output_path": None, "markdown": "", "error": "Vorgang durch Benutzer abgebrochen."}
            
        # Progress und Log weiterleiten
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

    worker.join(timeout=60)

    if not result_queue.empty():
        return result_queue.get()
    return {"success": False, "output_path": None, "markdown": "", "error": "Worker-Prozess ohne Ergebnis beendet"}


# ── Worker-Funktion ───────────────────────────────────────────────────────────

def run_ocr_summary_worker(
    source_path: Path,
    settings: dict,
    result_queue,
    progress_queue,
    log_queue,
    preview_queue,
) -> None:
    # WICHTIG: Einstellungen neu laden (für Multiprocessing-Spawn)
    from core.settings_manager import get_settings
    get_settings()

    """
    Worker-Funktion — läuft im separaten Prozess.
    Vollständige OCR-Pipeline: Vorbereitung → Docling → Analysis → Writing → Assembly.
    """
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
        _run_ocr_pipeline(source_path, settings, _progress, _log, result_queue, preview_queue)
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"❌ OCR-Worker Unerwarteter Fehler: {e}\n{tb}")
        result_queue.put({
            "success": False,
            "output_path": None,
            "markdown": "",
            "error": str(e),
        })


def _run_ocr_pipeline(
    source_path: Path,
    settings: dict,
    progress: callable,
    log: callable,
    result_queue,
    preview_queue,
) -> None:
    """Interne Pipeline-Implementierung."""
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
        PageGroup, OCRFileAnalysis,
    )
    from rag.tools import ToolExecutor
    from rag.ocr_tools import OCRToolExecutor
    from rag.augmenter import resolve_rag_image_tags

    # ── Einstellungen auslesen ──────────────────────────────────────────────
    detail_level     = settings.get("detail_level", 80)
    rag_store_name   = settings.get("rag_store_name", None)
    web_enabled      = settings.get("web_search_enabled", False)
    output_format    = settings.get("output_format", "pdf")
    writing_dpi      = settings.get("ocr_writing_dpi", config.OCR_WRITING_DPI)
    max_pages_write  = settings.get("ocr_max_pages_per_call", config.OCR_MAX_PAGES_PER_WRITING_CALL)
    project_name     = settings.get("project_name", source_path.stem)

    detail_desc = "umfassend und ausführlich" if detail_level > 60 else "kompakt und präzise"

    # ═════════════════════════════════════════════════════════════
    # PHASE A — VORBEREITUNG
    # ═════════════════════════════════════════════════════════════
    progress(2, "Prüfe LM Studio...")
    if not ensure_lm_studio_running():
        result_queue.put({"success": False, "output_path": None, "markdown": "",
                          "error": "LM Studio nicht erreichbar"})
        return

    progress(4, "Bereite PDF vor...")
    repaired = repair_pdf_if_needed(source_path)
    work_path = repaired  # kann original oder repaired_{name}.pdf sein

    page_count = get_page_count(work_path)
    if page_count == 0:
        result_queue.put({"success": False, "output_path": None, "markdown": "",
                          "error": f"PDF konnte nicht gelesen werden: {source_path.name}"})
        return

    log(f"📄 {source_path.name} — {page_count} Seiten")

    output_dir = Path(config.CENTRAL_OUTPUT_DIR) / f"{source_path.stem}_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    client = create_openai_client()

    # ═════════════════════════════════════════════════════════════
    # PHASE B — DOCLING FIGUREN-EXTRAKTION
    # ═════════════════════════════════════════════════════════════
    progress(8, "Extrahiere Abbildungen (Docling)...")
    log("🔍 Starte Docling Figuren-Extraktion...")
    deduplicator = AdvancedDeduplicator()
    all_figures = extract_figures_from_pdf(work_path, figures_dir, deduplicator)
    log(f"   ✅ {len(all_figures)} Abbildungen extrahiert")

    # ═════════════════════════════════════════════════════════════
    # PHASE C — ANALYSIS PASS
    # ═════════════════════════════════════════════════════════════
    progress(12, "Prüfe Analysis-Cache...")

    # Cache-Hit?
    cached = load_analysis_cache(work_path)
    if cached is not None:
        page_groups = cached.page_groups
        log(f"   ✅ Cache-Hit: {len(page_groups)} Themengruppen")
        # Figuren den Gruppen zuordnen (neu, da Figuren nicht im Cache stecken müssen)
        for group in page_groups:
            group.figures = get_figures_for_pages(all_figures, group.pages)
    else:
        progress(14, "Lade Vision-Modell für Analyse...")
        v_mod = settings.get("vision_model", config.VISION_MODEL_LOAD)
        vision_model_id = switch_model(client, v_mod, v_mod)
        log(f"   ✅ Modell: {vision_model_id}")

        progress(16, "Analysiere Themenstruktur...")
        log(f"📊 Analysis-Pass: {page_count} Seiten in Batches à {config.OCR_MAX_PAGES_PER_ANALYSIS_BATCH}")

        thumbnails = render_all_pages_as_thumbnails(work_path, dpi=config.OCR_ANALYSIS_DPI)
        if not thumbnails:
            log("   ⚠️ Rendering fehlgeschlagen — nutze Fallback-Gruppen")
            page_groups = _make_fallback_groups(work_path, page_count, config.OCR_FALLBACK_GROUP_SIZE)
        else:
            # Detect overarching lecture subject once before analysis batches so
            # each batch can contextualise its main_topics accordingly.
            document_topic = _detect_document_topic(thumbnails, client, vision_model_id)
            if document_topic:
                log(f"   🎯 Vorlesungsthema erkannt: {document_topic}")

            all_page_data: list[dict] = []
            batch_size = config.OCR_MAX_PAGES_PER_ANALYSIS_BATCH
            batches = [thumbnails[i:i + batch_size] for i in range(0, len(thumbnails), batch_size)]

            prev_main_topic = ""
            seen_analysis_topics: list[str] = []  # unique topics seen so far in analysis
            for batch_idx, batch in enumerate(batches):
                start_page = batch_idx * batch_size
                pct = 16 + int((batch_idx / len(batches)) * 14)
                progress(pct, f"Analyse Batch {batch_idx + 1}/{len(batches)}...")
                log(f"   🔎 Analyse Batch {batch_idx + 1}/{len(batches)} (Seiten {start_page + 1}–{start_page + len(batch)})")

                batch_data = _run_analysis_batch(
                    batch, start_page, client, vision_model_id,
                    prev_context=prev_main_topic,
                    recent_topics=seen_analysis_topics,
                    document_topic=document_topic,
                    batch_nr=batch_idx + 1,
                    total_batches=len(batches),
                )
                if batch_data:
                    all_page_data.extend(batch_data)
                    prev_main_topic = batch_data[-1].get("main_topic", "")
                    # Update unique topic list for next batch's context
                    for entry in batch_data:
                        t = entry.get("main_topic", "")
                        if t and t not in seen_analysis_topics and t not in ("Titelseite", "Gliederung"):
                            seen_analysis_topics.append(t)
                else:
                    # Fallback für diesen Batch — generische Themengruppen
                    fallback_entries = []
                    for i in range(len(batch)):
                        fallback_entries.append({
                            "page": start_page + i,
                            "main_topic": f"Abschnitt {(start_page + i) // config.OCR_FALLBACK_GROUP_SIZE + 1}",
                            "sub_topic": "",
                            "new_main_topic": (i == 0),
                        })
                    all_page_data.extend(fallback_entries)
                    # prev_main_topic auch im Fallback aktualisieren → nächster Batch
                    # bekommt korrekte Kontinuität (auch wenn Name generisch ist)
                    prev_main_topic = fallback_entries[-1]["main_topic"] if fallback_entries else prev_main_topic

            page_groups = _build_page_groups(all_page_data, work_path, max_pages_write)
            log(f"   ✅ {len(page_groups)} Themengruppen erkannt")

        # Figuren den Gruppen zuordnen
        for group in page_groups:
            group.figures = get_figures_for_pages(all_figures, group.pages)

        # Cache speichern
        analysis = OCRFileAnalysis(
            file_path=work_path,
            page_count=page_count,
            page_groups=page_groups,
        )
        save_analysis_cache(analysis)

    # ═════════════════════════════════════════════════════════════
    # PHASE D — WRITING PASS
    # ═════════════════════════════════════════════════════════════
    progress(32, "Lade Vision-Modell für Writing-Pass...")
    v_mod = settings.get("vision_model", config.VISION_MODEL_LOAD)
    vision_model_id = switch_model(client, v_mod, v_mod)

    base_executor = ToolExecutor(rag_store_name, log_callback=log, model_id=vision_model_id)
    sections: list[str] = []
    rolling_context = ""
    prev_main_topic = ""
    prev_sub_topic = ""
    global_inserted_ids: set[str] = set()  # dokumentweite Bildsperre
    sections_since_digest = 0             # Zähler für Kontext-Digest
    global_digest = ""                    # aktueller Digest vergangener Abschnitte
    seen_main_topics: list[str] = []      # deduplizierte Main-Topic-Liste für Rolling-Kontext

    total_groups = len(page_groups)
    writing_start_pct = 35
    writing_end_pct = 88

    for group_idx, group in enumerate(page_groups):
        pct = writing_start_pct + int((group_idx / max(total_groups, 1)) * (writing_end_pct - writing_start_pct))
        progress(pct, f"Schreibe Abschnitt {group_idx + 1}/{total_groups}: {group.main_topic}")

        # Skip-Topics überspringen
        if group.main_topic.lower().strip() in config.OCR_SKIP_TOPICS:
            log(f"   ⏭️ Übersprungen: {group.main_topic}")
            continue

        log(f"✍️ [{group_idx + 1}/{total_groups}] {group.main_topic} — {group.sub_topic} ({len(group.pages)} Seiten, {len(group.figures)} Abbildungen)")

        # Seiten rendern
        page_images = render_pages(work_path, group.pages, dpi=writing_dpi)
        if not page_images:
            log(f"   ⚠️ Rendering fehlgeschlagen für Gruppe {group_idx + 1}, überspringe")
            continue

        # OCRToolExecutor für diese Gruppe
        ocr_executor = OCRToolExecutor(
            figures=group.figures,
            output_figures_dir=output_dir / "images",
            base_executor=base_executor,
            log_callback=log,
            web_search_enabled=web_enabled,
        )
        # Globale Bildsperre übergeben (verhindert gleiche Bilder in verschiedenen Abschnitten)
        ocr_executor.set_global_inserted_ids(global_inserted_ids)
        base_executor.reset_section_counters()

        tools = ocr_executor.get_all_tools()

        # System-Prompt
        context_block = ""
        if rolling_context and prev_main_topic:
            # Kompakte Topic-Liste (max. 4 einzigartige Main-Topics) + Text-Ende
            topics_line = ""
            if seen_main_topics:
                topics_line = "Bereits abgehandelt: " + ", ".join(seen_main_topics[-4:]) + "\n"
            context_block = (
                f"VORHERIGER KONTEXT:\n"
                f"{topics_line}"
                f"Letztes Thema: {prev_main_topic} — {prev_sub_topic}\n"
                f"Ende letzter Abschnitt: ...{rolling_context[-300:]}\n\n"
            )

        # Globaler Digest: gibt dem Writer Überblick über bisherige Themen
        if global_digest:
            context_block += (
                f"BISHER BEHANDELTE THEMEN (Überblick):\n{global_digest}\n\n"
            )

        # is_new_main_topic-Hinweis: gibt dem Writer wichtigen strukturellen Kontext
        if group.is_new_main_topic:
            context_block += (
                "STRUKTURHINWEIS: Dies ist der Beginn eines NEUEN Hauptthemas. "
                "Starte mit einer kurzen thematischen Einleitung bevor du in die Details gehst.\n\n"
            )
        elif prev_main_topic:
            context_block += (
                f"STRUKTURHINWEIS: Dieses Thema ist eine Fortsetzung / ein Unterthema von "
                f"'{prev_main_topic}'. Stelle einen fließenden inhaltlichen Anschluss her.\n\n"
            )
        figure_list_text = build_figure_list_text(group.figures, ocr_executor.figure_aliases)
        figure_block = f"{figure_list_text}\n\n" if figure_list_text else ""

        from core.page_renderer import get_writing_cache_key, load_writing_cache, save_writing_cache
        
        # 1. Cache-Check
        batch_page_info = [(source_path.name, pno) for pno in group.pages]
        c_key = get_writing_cache_key(
            main_topic=group.main_topic,
            sub_topic=group.sub_topic,
            page_info=batch_page_info,
            model_id=vision_model_id,
            detail_level=detail_level
        )
        cached_text = load_writing_cache(c_key)
        
        if cached_text:
            log(f"   ♻️ Cache-Treffer für: {group.main_topic} — {group.sub_topic}")
            section_text = cached_text
        else:
            system_prompt = get_ocr_writing_prompt(
                detail_level=detail_level,
                detail_desc=detail_desc,
                context_block=context_block,
                figure_block=figure_block,
                main_topic=group.main_topic,
                sub_topic=group.sub_topic or "Allgemein"
            )

            # User-Message (Bilder + Aufgabe)
            user_content = _build_user_message_content(
                page_images=page_images,
                pages=group.pages,
                main_topic=group.main_topic,
                sub_topic=group.sub_topic,
                figures=group.figures,
                figure_aliases=ocr_executor.figure_aliases,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # Agentic Writing Call
            if tools and config.AGENT_TOOLS_ENABLED:
                section_text, tool_log = agentic_chat_completion(
                    client=client,
                    model=vision_model_id,
                    messages=messages,
                    tools=tools,
                    tool_executor=ocr_executor.execute,
                    log_callback=log,
                )
                if tool_log:
                    log(f"   🔧 {len(tool_log)} Tool-Calls in diesem Abschnitt")
            else:
                section_text = robust_chat_completion(
                    client=client,
                    model=vision_model_id,
                    messages=messages,
                )

            # ── Explicit memory release ───────────────────────────────────────
            # page_images (rendered PIL pages) and messages (base64-encoded
            # images) can be 20–100 MB per section. Release them immediately
            # after the LLM call to prevent unbounded RAM growth over long runs.
            for _img in page_images:
                try:
                    _img.close()
                except Exception:
                    pass
            del page_images, messages, user_content
            gc.collect()

            section_text = clean_llm_markdown_output(section_text)
            save_writing_cache(c_key, section_text)

        if section_text and not section_text.startswith("[FEHLER"):
            sections.append(section_text)
            # Globale Bildsperre aktualisieren
            global_inserted_ids.update(ocr_executor.get_global_inserted_ids())
            # Rolling Context = ENDE des letzten Abschnitts (für nahtlose Übergänge)
            rolling_context = section_text[-config.OCR_ROLLING_CONTEXT_CHARS:]
            prev_main_topic = group.main_topic
            prev_sub_topic = group.sub_topic
            # Deduplizierte Topic-Liste aktualisieren
            if group.main_topic and group.main_topic not in seen_main_topics:
                seen_main_topics.append(group.main_topic)
            sections_since_digest += 1

            # Kontext-Digest: alle OCR_DIGEST_INTERVAL Abschnitte eine globale
            # Zusammenfassung der bisher behandelten Themen erzeugen.
            if sections_since_digest >= config.OCR_DIGEST_INTERVAL:
                sections_since_digest = 0
                try:
                    # Gleitendes Fenster: nur die letzten OCR_DIGEST_INTERVAL*2 Abschnitte.
                    # Verhindert, dass der Digest-Input mit zunehmender Dokumentlänge
                    # unbegrenzt wächst und das Kontext-Window belastet.
                    window_start = max(0, group_idx + 1 - config.OCR_DIGEST_INTERVAL * 2)
                    digest_input = "\n".join(
                        f"- {g.main_topic}" + (f" / {g.sub_topic}" if g.sub_topic else "")
                        for g in page_groups[window_start:group_idx + 1]
                        if g.main_topic.lower().strip() not in config.OCR_SKIP_TOPICS
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
                    log(f"   📋 Kontext-Digest aktualisiert ({group_idx + 1} Abschnitte)")
                except Exception:
                    pass  # Digest-Fehler ist unkritisch
            
            # --- Progressive Preview ---
            temp_md = assemble_final_markdown(
                parts=sections,
                title=f"{project_name} (In Bearbeitung...)",
                detail_level=detail_level,
                source_files=[source_path.name],
            )
            preview_queue.put(temp_md)
        else:
            log(f"   ⚠️ Abschnitt leer oder Fehler: {section_text[:80]}")

    # ═════════════════════════════════════════════════════════════
    # PHASE E — ASSEMBLY & OUTPUT
    # ═════════════════════════════════════════════════════════════
    progress(90, "Erstelle finales Dokument...")
    log("📝 Assembling Markdown...")

    if not sections:
        result_queue.put({"success": False, "output_path": None, "markdown": "",
                          "error": "Kein Inhalt erzeugt — alle Abschnitte leer"})
        return

    # Buchtitel via LLM generieren (analog zu ocr_synthesis)
    import re as _re
    topic_names = list(dict.fromkeys(g.main_topic for g in page_groups if g.main_topic))[:6]
    topics_str = "; ".join(topic_names)
    try:
        title_prompt = (
            f"Erstelle einen kurzen akademischen Titel für ein medizinisches Lernskript mit diesen Themen:\n"
            f"{topics_str}\n"
            "Antworte NUR mit dem Titel (keine Anführungszeichen, kein Punkt am Ende)."
        )
        final_title = robust_chat_completion(
            client, vision_model_id,
            [{"role": "user", "content": title_prompt}],
            max_tokens=config.UTILITY_MAX_TOKENS,
            temperature=0.1,
        ).strip().replace('"', "").replace("'", "")
        if not final_title:
            final_title = project_name
    except Exception:
        final_title = project_name
    log(f"   📖 Titel: {final_title}")
    # Umlaut-Transkription vor Sonderzeichen-Entfernung (ä→ae etc.)
    _st = final_title
    for _src, _dst in [("ä","ae"),("ö","oe"),("ü","ue"),("Ä","ae"),("Ö","oe"),("Ü","ue"),("ß","ss")]:
        _st = _st.replace(_src, _dst)
    safe_title = _re.sub(r"[^\w\s\-]", "", _st).strip()
    safe_title = _re.sub(r"\s+", "_", safe_title)[:80] or project_name

    final_markdown = assemble_final_markdown(
        parts=sections,
        title=final_title,
        detail_level=detail_level,
        source_files=[source_path.name],
    )
    
    # RAG-Image-Tags auflösen (falls Bilder via RAG eingebunden wurden)
    final_markdown = resolve_rag_image_tags(final_markdown, output_dir / "images")

    # Markdown speichern
    md_path = output_dir / f"{safe_title}.md"
    save_markdown(final_markdown, md_path)

    # PDF erzeugen
    pdf_path = output_dir / f"{safe_title}.pdf"
    if output_format in ("pdf", "both"):
        progress(94, "Erzeuge PDF...")
        success_pdf = markdown_to_pdf(final_markdown, pdf_path, base_url=output_dir, detail_level=detail_level)
        if success_pdf:
            log(f"   ✅ PDF: {pdf_path}")
        else:
            log("   ⚠️ PDF-Erzeugung fehlgeschlagen, Markdown vorhanden")

    output_path = pdf_path if (output_format in ("pdf", "both") and pdf_path.exists()) else md_path
    progress(100, "Fertig!")
    log(f"✅ OCR-Verarbeitung abgeschlossen: {output_path}")

    # Optional Evidence PDF — lists every topic group with its processed page numbers.
    # Since OCR mode works image-based, no raw source text is available, only page references.
    if settings.get("evidence_pdf", False):
        log("🔎 Erstelle Evidence-PDF (Quellennachweis)...")
        evidence_md_parts = [
            f"# Quellen-Nachweis zu: {final_title}",
            "\n> Dieser Report zeigt die im OCR-Modus verarbeiteten Themen-Blöcke und Seiten.\n"
            "> Da die Verarbeitung visuell (als Bild) stattfand, liegt kein nativer Roh-Text vor.\n",
        ]
        evidence_md_parts.append(f"## Datei: {source_path.name}")
        for grp in page_groups:
            pages_str = ", ".join(str(p + 1) for p in grp.pages)
            evidence_md_parts.append(f"### {grp.main_topic} — {grp.sub_topic}")
            evidence_md_parts.append(f"**Verarbeitete Seiten:** {pages_str}\n")
            evidence_md_parts.append(f"*[Inhalte wurden als Bild-Kontext an das Vision-Modell übergeben.]*\n")
        ev_pdf_path = output_dir / f"{safe_title}_Quellen.pdf"
        ok_ev = markdown_to_pdf(
            "\n".join(evidence_md_parts), ev_pdf_path,
            base_url=output_dir, detail_level=detail_level,
        )
        if ok_ev:
            log(f"✅ Evidence-PDF: {ev_pdf_path}")
        else:
            ev_md_path = output_dir / f"{safe_title}_Quellen.md"
            save_markdown("\n".join(evidence_md_parts), ev_md_path)

    # Optional Anki export — creates simple flashcards from bold terms in the written sections
    if settings.get("anki_export", False):
        try:
            from modes.synthesis import _try_anki_export
            _try_anki_export(sections, output_dir, final_title, log)
        except Exception as e:
            log(f"⚠️ Anki-Export Fehler: {e}")

    # Delete temporary repaired PDF if one was created
    if repaired != source_path and repaired.exists():
        try:
            repaired.unlink()
        except Exception:
            pass

    result_queue.put({
        "success": True,
        "output_path": output_path,
        "markdown": final_markdown,
        "error": "",
    })


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _run_analysis_batch(
    thumbnails: list,
    start_page: int,
    client,
    model_id: str,
    prev_context: str = "",
    recent_topics: list[str] | None = None,
    document_topic: str = "",
    batch_nr: int = 0,
    total_batches: int = 0,
) -> list[dict]:
    """
    Sends a batch of thumbnails to the vision LLM for topic analysis.
    Returns list[dict], empty on error.

    prev_context    – main_topic string of the last page in the preceding batch.
    recent_topics   – ordered list of the last N unique main_topics seen so far.
    document_topic  – overarching subject detected by _detect_document_topic().
    batch_nr        – 1-based index of this batch within the file.
    total_batches   – total number of batches for this file.
    """
    from core.llm_client import robust_chat_completion, extract_json_robust
    from core.image_utils import pil_to_base64_jpeg

    n = len(thumbnails)
    end_page = start_page + n - 1

    content: list[dict] = []
    for i, thumb in enumerate(thumbnails):
        page_num = start_page + i
        b64 = pil_to_base64_jpeg(thumb, quality=72)
        # Close the PIL thumbnail immediately after encoding — the caller owns
        # the list and will keep all thumbnails in RAM otherwise.
        thumb.close()
        content.append({
            "type": "text",
            "text": f"--- Folie {page_num} ---",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": b64, "detail": "low"},
        })

    # ── Prompt placeholder strings ────────────────────────────────────────────

    # {topic_line}: suffix appended to the role line, e.g. " zum Thema Rheumatologie"
    topic_line = f" zum Thema {document_topic}" if document_topic else ""

    # {batch_line}: tells the LLM which chunk of the lecture this is
    if total_batches > 1 and batch_nr > 0:
        batch_line = f"Du bearbeitest Abschnitt {batch_nr} von {total_batches} dieser Vorlesung."
    else:
        batch_line = ""

    # {topic_forbidden_line}: dynamic warning not to re-use the broad subject name
    if document_topic:
        topic_forbidden_line = (
            f"\n⚠️ WICHTIG: Das übergeordnete Fachgebiet dieser Vorlesung ist "
            f"'{document_topic}'. Verwende diesen Begriff NIEMALS als main_topic — "
            f"gehe eine Ebene tiefer und nutze spezifischere klinische Kategorien!"
        )
    else:
        topic_forbidden_line = ""

    # ── Orientation context (recent topics + previous batch continuation) ──
    ctx_parts: list[str] = []
    if recent_topics:
        # Show at most the last 4 unique topics so the LLM can avoid near-duplicates.
        # Framing is explicit: these are PAST topics to orient by, not to re-use.
        topics_str = ", ".join(recent_topics[-4:])
        ctx_parts.append(
            f"Bereits abgehandelte Hauptthemen (zur Orientierung — NICHT wiederholen, "
            f"nur als Referenz): {topics_str}."
        )
    if prev_context:
        ctx_parts.append(
            f"Das Hauptthema der unmittelbar vorhergehenden Folie war: '{prev_context}'. "
            f"Führe dieses Thema weiter, sofern kein offensichtlicher Themenwechsel erkennbar ist."
        )
    ctx_str = ("\nWICHTIGER KONTEXT:\n" + "\n".join(ctx_parts) + "\n") if ctx_parts else ""

    content.append({
        "type": "text",
        "text": _ANALYSIS_PROMPT_TEMPLATE.format(
            n=n, start=start_page, end=end_page,
            topic_line=topic_line,
            batch_line=batch_line,
            topic_forbidden_line=topic_forbidden_line,
            prev_context=ctx_str,
        ),
    })

    messages = [
        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    response = robust_chat_completion(
        client=client,
        model=model_id,
        messages=messages,
        max_tokens=config.ANALYSIS_MAX_TOKENS,
        temperature=0.1,
    )
    # Release the large messages list (contains base64-encoded thumbnails).
    del messages, content

    if response.startswith("[FEHLER"):
        print(f"   ⚠️ Analysis-Batch Fehler: {response}")
        return []

    # ── Robuste Response-Vorverarbeitung ─────────────────────────────────────
    # 1. Explizit Code-Fences entfernen (```json, ```, ~~~)
    cleaned = re.sub(r"```(?:json)?\s*", "", response)
    cleaned = re.sub(r"```", "", cleaned).strip()
    # 2. Führenden Prosa-Text vor dem ersten [ entfernen
    #    (Modell schreibt manchmal "Hier ist das Ergebnis:\n[...")
    bracket_start = cleaned.find("[")
    if bracket_start > 0:
        cleaned = cleaned[bracket_start:]
    # 3. Trailing Text nach dem letzten ] entfernen
    bracket_end = cleaned.rfind("]")
    if bracket_end != -1 and bracket_end < len(cleaned) - 1:
        cleaned = cleaned[:bracket_end + 1]

    parsed = extract_json_robust(cleaned)
    if not isinstance(parsed, list):
        # Letzter Versuch: Original-Response (nicht cleaned) probieren
        parsed = extract_json_robust(response)

    if not isinstance(parsed, list):
        print(f"   ⚠️ Analysis-JSON ungültig, nutze Fallback")
        print(f"   🔍 Response-Preview: {response[:300].replace(chr(10), ' ')!r}")
        return []

    # Seitennummern validieren (müssen aufsteigend und im Bereich sein)
    valid = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        page = item.get("page", -1)
        if isinstance(page, int) and start_page <= page <= end_page:
            valid.append({
                "page": page,
                "main_topic": str(item.get("main_topic", "Unbekannt"))[:80],
                "sub_topic": str(item.get("sub_topic", ""))[:80],
                "new_main_topic": bool(item.get("new_main_topic", False)),
            })
    return valid


def _build_page_groups(
    page_analysis: list[dict],
    source_path: Path,
    max_pages_per_group: int,
) -> list["PageGroup"]:
    """
    Konvertiert Seiten-Analyse zu PageGroup-Objekten.
    Seiten werden nach Thema gruppiert, max_pages_per_group begrenzt die Größe.
    """
    from core.page_renderer import PageGroup

    if not page_analysis:
        return []

    page_analysis_sorted = sorted(page_analysis, key=lambda x: x["page"])

    groups: list[PageGroup] = []
    current_pages: list[int] = []
    current_topic = ""
    current_sub = ""
    is_new_main = True

    for entry in page_analysis_sorted:
        page_0idx = entry["page"]  # ist bereits 0-indexed (so wie wir es im Prompt angegeben haben)
        topic = entry["main_topic"] or "Unbekannt"
        sub = entry["sub_topic"] or ""
        new_main = entry["new_main_topic"]

        # FIX: Wenn die KI signalisiert, dass es kein neues Thema ist, erzwingen wir
        # den aktuellen Titel, um String-Abweichungen ("Topic" vs "Topic - Klinik") zu ignorieren!
        if not new_main and current_topic:
            topic = current_topic

        topic_changed = (topic != current_topic) or new_main
        group_full = len(current_pages) >= max_pages_per_group

        if (topic_changed or group_full) and current_pages:
            groups.append(PageGroup(
                file_path=source_path,
                pages=current_pages,
                main_topic=current_topic,
                sub_topic=current_sub,
                is_new_main_topic=is_new_main,
            ))
            current_pages = []
            # is_new_main direkt aus LLM-Bewertung (new_main) — nicht aus String-Vergleich.
            # topic_changed wird nur für die Split-Entscheidung genutzt, nicht für den Flag.
            is_new_main = new_main

        current_pages.append(page_0idx)
        current_topic = topic
        current_sub = sub

    # Letzten Block abschließen
    if current_pages:
        groups.append(PageGroup(
            file_path=source_path,
            pages=current_pages,
            main_topic=current_topic,
            sub_topic=current_sub,
            is_new_main_topic=is_new_main,
        ))

    return groups


def _make_fallback_groups(
    source_path: Path,
    page_count: int,
    group_size: int,
) -> list["PageGroup"]:
    """Erstellt feste Gruppen wenn Analysis fehlschlägt."""
    from core.page_renderer import PageGroup

    groups = []
    for start in range(0, page_count, group_size):
        end = min(start + group_size, page_count)
        groups.append(PageGroup(
            file_path=source_path,
            pages=list(range(start, end)),
            main_topic=f"Abschnitt {start // group_size + 1}",
            sub_topic="",
            is_new_main_topic=True,
        ))
    return groups


def _build_user_message_content(
    page_images: list,
    pages: list[int],
    main_topic: str,
    sub_topic: str,
    figures: list["DoclingFigure"] = None,
    figure_aliases: dict[str, str] = None,
) -> list[dict]:
    """Baut den multimodalen User-Message-Content für den Writing-Pass."""
    from core.image_utils import pil_to_base64_jpeg
    from PIL import Image

    content: list[dict] = []
    
    # 1. Vorlesungsfolien (Hauptbilder)
    # Note: caller closes the PIL images after this function returns, so we
    # only need the base64 string here — no additional close() needed.
    for i, img in enumerate(page_images):
        page_num = pages[i] if i < len(pages) else "?"
        b64 = pil_to_base64_jpeg(img)
        content.append({
            "type": "text",
            "text": f"--- Vorlesungsfolie {page_num} ---",
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

    content.append({
        "type": "text",
        "text": _WRITING_USER_TEMPLATE.format(
            main_topic=main_topic,
            sub_topic=sub_topic or "Allgemein",
            n=len(page_images),
        ),
    })
    return content
