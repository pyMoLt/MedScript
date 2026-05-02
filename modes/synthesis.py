# modes/synthesis.py — Mode 2: multiple files to synthesized textbook

# Mode 2: Multiple files → synthesized textbook (deep synthesis)

from __future__ import annotations
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
import markdown

import config
from core.llm_client import (
    create_openai_client, robust_chat_completion, agentic_chat_completion,
    switch_model, unload_model, get_embedding, extract_json_robust,
)
from core.pdf_parser import (
    repair_pdf_if_needed, build_docling_converter, ingest_pdf,
    get_item_context, is_image_in_margin,
)
from core.image_utils import AdvancedDeduplicator, pil_to_base64_jpeg, save_pil_image, get_image_hash
from core.text_utils import (
    is_text_junk, clean_llm_markdown_output, inject_markdown_headers,
)
from core.output import assemble_final_markdown, markdown_to_pdf, save_markdown
from core.cache import SimpleCache
from rag.tools import get_tools_for_run, ToolExecutor
from rag.augmenter import get_writer_tool_instruction, resolve_rag_image_tags


# ── Datenstrukturen ───────────────────────────────────────────────────────────

@dataclass
class RichChunk:
    id: str
    text: str
    source_file: str
    attached_images: list[dict] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class TopicBlock:
    id: str
    source_file: str
    title: str
    chunks: list[RichChunk] = field(default_factory=list)


@dataclass
class FileContainer:
    source_path: Path
    name: str
    topic_blocks: list[TopicBlock] = field(default_factory=list)
    major_topics: list[str] = field(default_factory=list)


# ── Kosinus-Ähnlichkeit ───────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ── Stil-Instruktionen ────────────────────────────────────────────────────────

def _get_style_instructions(detail_level: int) -> tuple[str, str]:
    """Gibt (style_instr, hybrid_instr) basierend auf Detailgrad zurück."""
    if detail_level >= 80:
        style_instr = (
            "OBERSTE PRIORITÄT: UMFASSENDE DETAILTREUE. Integriere JEDES Detail aus dem Folienstoff. "
            "Akademischer Stil. Fachbegriffe hervorheben (**Begriff**). "
            "Wichtige Merksätze als Blockquote (> ...). "
            "Mechanismen vollständig und Schritt-für-Schritt erklären. "
            "Tabellen für Vergleiche und Klassifikationen nutzen. "
            "Strukturiere den Inhalt mit Markdown-Überschriften (##, ###)."
        )
        hybrid_instr = ""
    elif detail_level >= 40:
        style_instr = (
            "PRIORITÄT: PRÄZISE & GUT LESBAR. "
            "Strukturiere logisch mit Markdown-Überschriften (##, ###)."
        )
        hybrid_instr = (
            "\n\nFORMAT-VORGABE (HYBRID-STIL — WICHTIG!):\n"
            "1. FLIESSTEXT: Nur für komplexe Zusammenhänge (Prozesse, Wirkmechanismen, Herleitungen).\n"
            "2. BULLETPOINTS: Für ALLES andere (Listen, Eigenschaften, Fakten, Symptome, Indikationen).\n"
            "3. REDUNDANZ: Fasse doppelte Informationen radikal zusammen. Sei präzise.\n"
            "4. Ziel: Hohe Informationsdichte, perfekt lesbar."
        )
    else:
        style_instr = "PRIORITÄT: NUR FACTS. Maximale Verdichtung."
        hybrid_instr = (
            "\n\nFORMAT-VORGABE (CHEAT-SHEET / ULTRA-CONDENSED):\n"
            "1. FAKTEN/LISTEN: Hierarchische Bulletpoints.\n"
            "2. PROZESSE: Kausalketten mit Pfeilen (Ursache → Prozess → Wirkung).\n"
            "3. STIL: Telegramm — Subjekt/Prädikat weglassen wo möglich.\n"
            "4. INHALT: Nur High-Yield: Definitionen, Parameter/Grenzwerte, Red Flags, Kernmechanismen.\n"
            "5. WEGLASSEN: Herleitungen, historische Aspekte, triviale Grundlagen."
        )
    return style_instr, hybrid_instr


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────

def process_deep_synthesis(
    source_files: list[Path],
    settings: dict,
    progress_callback: callable = None,
    log_callback: callable = None,
    cancel_event: threading.Event = None,
    preview_callback: callable = None,
) -> dict:
    """Hauptfunktion für Multi-Datei Deep Synthesis."""

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
    rag_store_name = settings.get("rag_store_name", None)
    output_format = settings.get("output_format", "pdf")
    web_enabled = settings.get("web_search_enabled", False)
    evidence_pdf = settings.get("evidence_pdf", False)
    anki_export = settings.get("anki_export", False)
    project_name = settings.get("project_name", "deep_synthesis")

    # WICHTIG: Einstellungen neu laden
    from core.settings_manager import get_settings
    get_settings()

    try:
        client = create_openai_client()
        output_dir = Path(config.CENTRAL_OUTPUT_DIR) / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        cache = SimpleCache(output_dir / ".vis_cache.json")

        containers: list[FileContainer] = []
        global_dedup = AdvancedDeduplicator()

        # ── PHASE 1: INGESTION & HARD BINDING ─────────────────────────────────
        check_cancel()
        log(f"📚 Deep Synthesis: {len(source_files)} Datei(en)")
        progress(2, "Starte Ingestion...")

        try:
            from docling_core.types.doc import TextItem, TableItem, PictureItem
        except ImportError:
            from docling.datamodel.base_models import TextItem, TableItem, PictureItem  # type: ignore

        for file_idx, pdf_path in enumerate(source_files):
            check_cancel()
            base_pct = 2 + int(file_idx / len(source_files) * 18)
            log(f"📄 Ingestion: {pdf_path.name}")
            progress(base_pct, f"Lese {pdf_path.name}...")

            working_path = repair_pdf_if_needed(pdf_path)
            try:
                converter = build_docling_converter()
                all_items, doc = ingest_pdf(working_path, converter)
            except Exception as e:
                log(f"❌❌❌ Docling-Fehler {pdf_path.name}: {e}")
                continue
            finally:
                if working_path != pdf_path and working_path.exists():
                    try:
                        working_path.unlink()
                    except Exception:
                        pass

            container = FileContainer(source_path=pdf_path, name=pdf_path.stem)
            current_block = TopicBlock(
                id=str(uuid.uuid4())[:8],
                source_file=pdf_path.name,
                title="Einleitung",
            )
            current_chunk = RichChunk(
                id=str(uuid.uuid4())[:8],
                text="",
                source_file=pdf_path.name,
            )
            img_idx_file = 0
            all_items_list = list(all_items)

            for item_idx, (item, level) in enumerate(all_items_list):
                if item_idx % 20 == 0:
                    check_cancel()

                if isinstance(item, TextItem):
                    text = item.text.strip()
                    label = str(getattr(item, "label", "")).lower()
                    if is_text_junk(text, label):
                        continue
                    is_header = "header" in label or "title" in label
                    text = inject_markdown_headers(text, label)

                    if is_header and current_chunk.text.strip():
                        current_block.chunks.append(current_chunk)
                        container.topic_blocks.append(current_block)
                        current_block = TopicBlock(
                            id=str(uuid.uuid4())[:8],
                            source_file=pdf_path.name,
                            title=text.lstrip("# ").strip(),
                        )
                        current_chunk = RichChunk(
                            id=str(uuid.uuid4())[:8], text="", source_file=pdf_path.name
                        )
                    else:
                        current_chunk.text += "\n\n" + text

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
                        if global_dedup.is_duplicate(pil_img, f"{pdf_path.stem}_{img_idx_file}"):
                            continue
                        img_fname = f"{pdf_path.stem}_{img_idx_file}.jpg"
                        img_path = images_dir / img_fname
                        save_pil_image(pil_img, img_path)
                        ctx = get_item_context(all_items_list, item_idx)
                        current_chunk.attached_images.append({
                            "path": str(img_path),
                            "id": f"{pdf_path.stem}_{img_idx_file}",
                            "desc": "",
                            "embedding": None,
                            "raw_context": ctx,
                        })
                        img_idx_file += 1
                    except Exception:
                        pass

            if current_chunk.text.strip():
                current_block.chunks.append(current_chunk)
            if current_block.chunks:
                container.topic_blocks.append(current_block)

            # ── Block-Merge: Kleinstblöcke zusammenführen ─────────────────────
            # Blöcke mit sehr wenig Text (Titelseiten, Agenda-Folien, etc.)
            # werden mit dem Folgeblock zusammengeführt, um Granularität zu reduzieren.
            MIN_BLOCK_CHARS = config.SYNTHESIS_MIN_BLOCK_CHARS
            merged_blocks: list[TopicBlock] = []
            carry: TopicBlock | None = None
            for blk in container.topic_blocks:
                if carry is not None:
                    # Chunks und Bilder des carry-Blocks in diesen Block übernehmen
                    blk.chunks = carry.chunks + blk.chunks
                    carry = None
                # Größe NACH dem Merge berechnen, damit zusammengeführte Blöcke
                # korrekt bewertet werden und nicht erneut in carry landen.
                block_text_len = sum(len(ch.text) for ch in blk.chunks)
                if block_text_len < MIN_BLOCK_CHARS and blk is not container.topic_blocks[-1]:
                    carry = blk  # zu kleiner Block → mit nächstem zusammenführen
                else:
                    merged_blocks.append(blk)
            # Falls letzter Block noch im carry hängt
            if carry is not None:
                if merged_blocks:
                    merged_blocks[-1].chunks.extend(carry.chunks)
                else:
                    merged_blocks.append(carry)
            container.topic_blocks = merged_blocks
            # ─────────────────────────────────────────────────────────────────

            containers.append(container)
            log(f"   ✅ {len(container.topic_blocks)} Blöcke (nach Merge), {img_idx_file} Bilder")

        # ── PHASE 2: VISUAL BRIDGE ────────────────────────────────────────────
        all_imgs = []
        for c in containers:
            for blk in c.topic_blocks:
                for chk in blk.chunks:
                    all_imgs.extend(chk.attached_images)

        vis_model_id = None
        if all_imgs:
            progress(22, "Vision-Modell lädt (Visual Bridge)...")
            log(f"👁️ Visual Bridge: {len(all_imgs)} Bilder beschriften...")
            v_mod = settings.get("vision_model", config.VISION_MODEL_LOAD)
            vis_model_id = switch_model(client, v_mod, v_mod)
            for i, img_data in enumerate(all_imgs):
                check_cancel()
                progress(22 + int(i / len(all_imgs) * 8), f"Bild {i+1}/{len(all_imgs)}...")
                cache_key = f"vb_{img_data['id']}"
                cached = cache.get(cache_key)
                if cached:
                    img_data["desc"] = cached
                    continue
                try:
                    from PIL import Image
                    pil_img = Image.open(img_data["path"])
                    b64 = pil_to_base64_jpeg(pil_img)
                    pil_img.close()   # release pixel buffer immediately after encoding
                    del pil_img
                    prompt = (
                        "Beschreibe dieses Bild aus einer medizinischen Vorlesungsfolie präzise und inhaltsdicht.\n"
                        f"Kontext: {img_data['raw_context'][:300]}\n"
                        f"Max {config.SYNTHESIS_VISUAL_BRIDGE_MAX_TOKENS} Tokens. Fließtext. "
                        "Nenne wichtige Fachbegriffe, beschriftete Elemente und Schlüsselkonzepte explizit — "
                        "diese werden für semantisches Matching genutzt."
                    )
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": b64}},
                    ]}]
                    desc = robust_chat_completion(
                        client, vis_model_id, messages,
                        max_tokens=config.SYNTHESIS_VISUAL_BRIDGE_MAX_TOKENS,
                    )
                    if desc.startswith("[FEHLER"):
                        desc = img_data["raw_context"][:200]
                    img_data["desc"] = desc
                    cache.set(cache_key, desc)
                    cache.save()
                except Exception:
                    img_data["desc"] = img_data["raw_context"][:200]
            if vis_model_id:
                unload_model(vis_model_id)

        # ── PHASE 3: EMBEDDING ────────────────────────────────────────────────
        progress(31, "Embedding-Modell lädt...")
        log("🔢 Berechne Embeddings...")
        emb_model_id = switch_model(client, config.EMBEDDING_MODEL_ID, config.EMBEDDING_MODEL_ID)

        all_chunks_flat = [chk for c in containers for blk in c.topic_blocks for chk in blk.chunks]
        for i, chk in enumerate(all_chunks_flat):
            if i % 10 == 0:
                check_cancel()
            if chk.text.strip():
                chk.embedding = get_embedding(client, chk.text[:1000], emb_model_id)
            for img_data in chk.attached_images:
                if img_data.get("desc"):
                    img_data["embedding"] = get_embedding(client, img_data["desc"], emb_model_id)

        # Embedding-Modell entladen bevor Text-Modell geladen wird
        unload_model(emb_model_id)

        # ── PHASE 4A: BLOCK-TITEL GENERIERUNG & HAUPTTHEMEN ──────────────────
        progress(40, "Text-Modell lädt (Strukturierung)...")
        log("🏷️  Generiere Block-Titel & analysiere Hauptthemen...")
        t_mod = settings.get("text_model", config.TEXT_MODEL_LOAD)
        text_model_id = switch_model(client, t_mod, t_mod)

        for container in containers:
            check_cancel()

            # LLM-Titel für jeden Block generieren
            for blk in container.topic_blocks:
                if not blk.chunks:
                    continue
                preview = " ".join(
                    ch.text[:200] for ch in blk.chunks[:2]
                )[:400].strip()
                if not preview:
                    continue
                label_prompt = (
                    "Gib diesem Vorlesungsabschnitt einen kurzen, prägnanten akademischen Titel "
                    "(2-6 Wörter, keine Nummerierung, kein Doppelpunkt am Ende).\n"
                    f"Textauszug:\n\"\"\"{preview}\"\"\"\n"
                    "Antworte NUR mit dem Titel."
                )
                new_title = robust_chat_completion(
                    client, text_model_id,
                    [{"role": "user", "content": label_prompt}],
                    max_tokens=config.UTILITY_MAX_TOKENS,
                )
                clean = new_title.strip().replace('"', "").replace("'", "")
                if clean:
                    blk.title = clean
                log(f"   🏷️  [{container.name}] → {blk.title}")

            # Major Topics aus LLM-Titeln extrahieren
            block_titles_str = "; ".join(blk.title for blk in container.topic_blocks[:10])
            topics_prompt = (
                f"Hier sind die Abschnitts-Titel einer Vorlesungs-Datei:\n{block_titles_str}\n\n"
                "Identifiziere die 1-3 dominierenden medizinischen Hauptthemen dieser Datei.\n"
                "Beispiele: 'Affektive Störungen', 'KHK & Herzinsuffizienz', 'Gastrointestinale Tumoren'.\n"
                "Antworte als JSON-Liste: [\"Thema 1\", \"Thema 2\"]"
            )
            raw = robust_chat_completion(
                client, text_model_id,
                [{"role": "user", "content": topics_prompt}],
                max_tokens=config.ANALYSIS_MAX_TOKENS,
            )
            topics = extract_json_robust(raw)
            container.major_topics = topics if isinstance(topics, list) else [container.name]
            log(f"   📌 Hauptthemen [{container.name}]: {container.major_topics}")

        # ── PHASE 4B: GLOBAL ARCHITECT (Kapitelstruktur) ──────────────────────
        progress(50, "Kapitelstruktur planen...")
        log("🏗️ Global Architect plant Kapitelstruktur...")

        # Übersicht mit echten Themennamen (nicht Dateinamen)
        files_overview = "\n".join(
            f"QUELLE '{chr(65+i)}' (Inhalt: {', '.join(c.major_topics[:3])})"
            for i, c in enumerate(containers)
        )
        arch_prompt = (
            "Du bist Chef-Redakteur für ein akademisches Lehrbuch.\n"
            f"Verfügbare Quellen:\n{files_overview}\n\n"
            "AUFGABE: Erstelle eine logische Kapitel-Struktur.\n"
            "- Jedes Kapitel basiert auf einer oder mehreren Quellen (IDs A, B, C...).\n"
            "- BÜNDELUNG (WICHTIG): Fasse verwandte Krankheitsbilder ZWINGEND unter einem gemeinsamen Oberthema zusammen.\n"
            "  Beispiel: Statt 'Rheumatoide Arthritis' und 'Lupus' als eigene Kapitel zu führen, erstelle ein Kapitel 'Entzündliche Systemerkrankungen'.\n"
            "- Bündele verwandte Themen aus verschiedenen Quellen ZWINGEND in EINEM gemeinsamen Kapitel. Vermeide Redundanz.\n"
            "- Didaktische Reihenfolge: Grundlagen → Spezielles.\n"
            "- WICHTIG: Erstelle die Titel OHNE eigene Nummerierung oder Präfixe. Schreibe nur den reinen Fachbegriff!\n"
            "Antworte als JSON:\n"
            '[{"title": "Affektive Störungen", "ids": ["A"], "focus": "Klinik, Diagnostik, Therapie"}]'
        )
        raw_plan = robust_chat_completion(
            client, text_model_id,
            [{"role": "user", "content": arch_prompt}],
            max_tokens=config.ANALYSIS_MAX_TOKENS,
        )
        chapter_plan = extract_json_robust(raw_plan)
        if not isinstance(chapter_plan, list) or not chapter_plan:
            log("   ⚠️ Master-Plan fehlgeschlagen — Fallback: 1 Datei = 1 Kapitel")
            chapter_plan = [
                {
                    "title": c.major_topics[0] if c.major_topics else c.name,
                    "ids": [chr(65 + i)],
                    "focus": "Gesamter Inhalt",
                }
                for i, c in enumerate(containers)
            ]

        # Buchtitel via LLM generieren
        chapter_titles_str = "; ".join(c.get("title", "Kapitel") for c in chapter_plan)
        title_prompt = (
            f"Erstelle einen kurzen akademischen Buchtitel für ein Lehrbuch mit diesen Kapiteln:\n"
            f"{chapter_titles_str}\n"
            "Antworte NUR mit dem Titel (keine Anführungszeichen, kein Punkt am Ende)."
        )
        final_book_title = robust_chat_completion(
            client, text_model_id,
            [{"role": "user", "content": title_prompt}],
            max_tokens=config.UTILITY_MAX_TOKENS,
        ).strip().replace('"', "").replace("'", "")
        if not final_book_title:
            final_book_title = project_name
        log(f"   📖 Buchtitel: {final_book_title}")
        log(f"   📋 {len(chapter_plan)} Kapitel geplant")

        # ── PHASE 4C: CONTENT WRITING (3-stufig) ─────────────────────────────
        progress(55, "Content-Writing startet...")
        log(f"✍️  Schreibe {len(chapter_plan)} Kapitel (3-stufig: Selektion → Struktur → Text)...")

        tools = get_tools_for_run(rag_store_name, web_enabled)
        executor = ToolExecutor(rag_store_name, log_callback, model_id=text_model_id) if tools else None
        tool_instr = get_writer_tool_instruction(rag_store_name, web_enabled)

        # Datei-ID → Container Mapping
        file_id_map = {chr(65 + i): c for i, c in enumerate(containers)}

        written_chapters: list[str] = []
        n_chapters = len(chapter_plan)

        for ch_idx, chapter in enumerate(chapter_plan):
            check_cancel()
            ch_pct = 55 + int(ch_idx / n_chapters * 27)
            ch_title = chapter.get("title", f"Kapitel {ch_idx + 1}")
            ch_focus = chapter.get("focus", "")
            ch_ids = chapter.get("ids", [])
            progress(ch_pct, f"Schreibe {ch_title}...")
            log(f"\n   📖 Kapitel {ch_idx + 1}: {ch_title}")

            # Kandidaten aus Quellen sammeln
            candidate_blocks: list[TopicBlock] = []
            for fid in ch_ids:
                cont = file_id_map.get(fid)
                if cont:
                    candidate_blocks.extend(cont.topic_blocks)

            if not candidate_blocks:
                log(f"   ⚠️ Keine Blöcke für Kapitel '{ch_title}' gefunden.")
                continue

            # ── SCHRITT 1: Block-Selektion ────────────────────────────────────
            selected_blocks: list[TopicBlock]
            if len(candidate_blocks) > 1 and ch_focus:
                short_id_map: dict[str, TopicBlock] = {}
                preview_lines: list[str] = []
                for idx, blk in enumerate(candidate_blocks):
                    s_id = f"B{idx}"
                    short_id_map[s_id] = blk
                    preview_lines.append(f"ID '{s_id}': {blk.title}")

                select_prompt = (
                    f"Wir schreiben das Kapitel '{ch_title}'.\n"
                    f"Fokus: {ch_focus}\n\n"
                    "Verfügbare Abschnitte:\n" + "\n".join(preview_lines) + "\n\n"
                    "Wähle NUR die Abschnitts-IDs die inhaltlich zu diesem Kapitel-Fokus gehören.\n"
                    "Ignoriere Abschnitte die ein völlig anderes Thema behandeln.\n"
                    "Antworte als JSON-Liste: [\"B0\", \"B2\"]"
                )
                sel_resp = robust_chat_completion(
                    client, text_model_id,
                    [{"role": "user", "content": select_prompt}],
                    max_tokens=config.ANALYSIS_MAX_TOKENS,
                )
                sel_ids_raw = extract_json_robust(sel_resp)
                if sel_ids_raw and isinstance(sel_ids_raw, list):
                    # Robuste ID-Extraktion: sucht das erste B\d+ Muster je Element
                    # Beispiele: "B0" → "B0", "ID 'B0'" → "B0", "Block B10" → "B10"
                    sel_ids_clean = []
                    for s in sel_ids_raw:
                        m = re.search(r'B\d+', str(s))
                        if m:
                            sel_ids_clean.append(m.group())
                    selected_blocks = [
                        short_id_map[s] for s in sel_ids_clean if s in short_id_map
                    ]
                else:
                    selected_blocks = candidate_blocks
                if not selected_blocks:
                    selected_blocks = candidate_blocks
            else:
                selected_blocks = candidate_blocks

            log(f"   📂 {len(selected_blocks)}/{len(candidate_blocks)} Blöcke ausgewählt")

            # Einmal berechnen — wird für Sub-Sektion-Planung und Transition-Hint genutzt
            previous_chapters = [c.get("title", "") for c in chapter_plan[:ch_idx]]

            # ── SCHRITT 2: Sub-Sektion-Planung ────────────────────────────────
            sub_sections: list[dict]
            if len(selected_blocks) > 1:
                sub_map: dict[str, TopicBlock] = {}
                sub_lines: list[str] = []
                for idx, blk in enumerate(selected_blocks):
                    s_id = f"S{idx}"
                    sub_map[s_id] = blk
                    sub_lines.append(f"ID '{s_id}': {blk.title}")

                prev_ch_hint = f"WICHTIG: Folgende Themen wurden bereits in vorherigen Kapiteln behandelt: {', '.join(previous_chapters)}.\nPlane keine Unterabschnitte zu diesen Themen, es sei denn es sind völlig neue Aspekte!\n\n" if previous_chapters else ""

                sub_prompt = (
                    f"Plane die Feinstruktur für das Kapitel '{ch_title}'.\n"
                    "Verfügbare Abschnitte:\n" + "\n".join(sub_lines) + "\n\n"
                    f"{prev_ch_hint}"
                    "Gruppiere ALLE Abschnitte in logische Unterabschnitte.\n"
                    "WICHTIG: Titel OHNE Nummerierung (z.B. 'Klinik', nicht '1.2 Klinik'). Verwende NIEMALS Präfixe wie 'Abschnitt'.\n"
                    "Ordne JEDE ID mindestens einem Unterabschnitt zu — lass nichts weg.\n"
                    "Antworte NUR als JSON:\n"
                    "[{\"title\": \"Ätiologie & Pathophysiologie\", \"ids\": [\"S0\", \"S1\"]}]"
                )
                sub_resp = robust_chat_completion(
                    client, text_model_id,
                    [{"role": "user", "content": sub_prompt}],
                    max_tokens=config.ANALYSIS_MAX_TOKENS,
                )
                sub_plan = extract_json_robust(sub_resp)
                if sub_plan and isinstance(sub_plan, list):
                    sub_sections = []
                    assigned_ids: set[str] = set()
                    for sec in sub_plan:
                        s_ids_raw = sec.get("ids", [])
                        s_blocks = [sub_map[s] for s in s_ids_raw if s in sub_map]
                        if s_blocks:
                            sub_sections.append({
                                "title": sec.get("title", "Abschnitt"),
                                "blocks": s_blocks,
                            })
                            assigned_ids.update(s_ids_raw)
                    # Nicht-zugeordnete Blöcke als letzten Abschnitt anhängen
                    leftover = [blk for sid, blk in sub_map.items() if sid not in assigned_ids]
                    if leftover:
                        sub_sections.append({"title": "Weitere Inhalte", "blocks": leftover})
                else:
                    sub_sections = [{"title": ch_title, "blocks": selected_blocks}]
            else:
                sub_sections = [{"title": ch_title, "blocks": selected_blocks}]

            log(f"   📑 {len(sub_sections)} Unterabschnitte geplant")

            # ── SCHRITT 3: Schreiben (je Unterabschnitt separat) ──────────────
            chapter_parts: list[str] = []
            last_section_end = ""
            last_section_full = ""  # vollständiger Text für Header-Extraktion
            style_instr, hybrid_instr = _get_style_instructions(detail_level)

            for sub_idx, sub in enumerate(sub_sections):
                check_cancel()
                sub_title = sub["title"]
                sub_blocks = sub["blocks"]

                full_txt = "\n\n".join(
                    f"[{blk.title}]\n"
                    + "\n".join(ch.text for ch in blk.chunks if ch.text.strip())
                    for blk in sub_blocks
                ).strip()
                if not full_txt:
                    continue

                # Transition-Hint: 400 Zeichen + letzter Unter-Header für besseren Kontext
                transition_hint = ""
                if previous_chapters:
                    transition_hint += f"\nINFO: Folgende Hauptthemen wurden bereits im Skript behandelt: {', '.join(previous_chapters)}. Fasse dich bei Wiederholungen extrem kurz."

                if last_section_end:
                    last_header_matches = list(re.finditer(r'###\s+(.+)', last_section_full))
                    last_header_title = last_header_matches[-1].group(1).strip() if last_header_matches else ""
                    if last_header_title:
                        transition_hint += (
                            f"\nINFO: Vorheriger Abschnitt — zuletzt behandeltes Unterthema: \"{last_header_title}\". "
                            f"Kontext-Ende: \"...{last_section_end}\". Stelle einen flüssigen Übergang her."
                        )
                    else:
                        transition_hint += (
                            f"\nINFO: Vorheriger Abschnitt endete mit: \"...{last_section_end}\". "
                            "Stelle einen flüssigen Übergang her."
                        )

                sec_num = f"{ch_idx + 1}.{sub_idx + 1}"

                # Bilder aus den Sub-Blöcken sammeln für multimodalen Writer-Call
                sub_images = []
                for blk in sub_blocks:
                    for chk in blk.chunks:
                        sub_images.extend(chk.attached_images)

                img_hint = ""
                if sub_images:
                    img_hint = (
                        "\n\nABBILDUNGEN: Zu diesem Abschnitt sind Abbildungen verfügbar (im User-Message). "
                        "Referenziere sie im Text mit \"wie in der Abbildung zu sehen\" oder "
                        "\"die Abbildung zeigt...\" an passender Stelle."
                    )

                focus_hint = f"\nKAPITEL-FOKUS: {ch_focus}" if ch_focus else ""
                system_prompt = (
                    f"Du bist Fachbuch-Autor für das Lehrbuch '{final_book_title}'.\n"
                    f"KAPITEL: {ch_title}{focus_hint}\n"
                    f"AKTUELLER ABSCHNITT (Thema): {sub_title}\n"
                    f"AUFGABE: {style_instr}{hybrid_instr}{transition_hint}{img_hint}\n\n"
                    f"{tool_instr}\n\n"
                    "SCHREIB-REGELN:\n"
                    f"1. Beginne exakt mit der Überschrift: ## {sec_num} {sub_title}\n"
                    "2. Vermeide allgemeine Einleitungen — steige DIREKT in den fachlichen Inhalt ein.\n"
                    "3. KONTINUITÄT: Wiederhole keine Informationen aus vorherigen Abschnitten.\n"
                    "4. TREUE: Erfinde NICHTS dazu. Folienstoff ist Pflicht-Input, Ergänzungen willkommen.\n"
                    "5. Keine Formeln in LaTeX — ausschließlich Klartext.\n"
                    "6. Wenn ein Tool-Ergebnis nicht relevant ist: ignoriere es, schreibe trotzdem weiter."
                )

                # User-Message: Folienstoff + optional Bilder als multimodale Inhalte
                if sub_images:
                    user_content: list[dict] = [{"type": "text", "text": f"FOLIENSTOFF:\n\n{full_txt}"}]
                    shown_imgs = 0
                    for img_data in sub_images:
                        if shown_imgs >= 6:
                            break
                        img_path = img_data.get("path", "")
                        if not img_path or not Path(img_path).exists():
                            continue
                        try:
                            from PIL import Image as _PILImage
                            _pil = _PILImage.open(img_path)
                            _b64 = pil_to_base64_jpeg(_pil)
                            _pil.close()   # release pixel buffer immediately
                            del _pil
                            _alias = img_data.get("id", f"Abb_{shown_imgs + 1}")
                            user_content.append({
                                "type": "text",
                                "text": f"\nAbbildung (ID: {_alias}):",
                            })
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": _b64, "detail": "low"},
                            })
                            shown_imgs += 1
                        except Exception:
                            pass
                    user_msg: dict = {"role": "user", "content": user_content}
                else:
                    user_msg = {"role": "user", "content": f"FOLIENSTOFF:\n\n{full_txt}"}

                messages = [
                    {"role": "system", "content": system_prompt},
                    user_msg,
                ]

                log(f"   ✍️  {sec_num} {sub_title}...")
                if tools and executor and config.AGENT_TOOLS_ENABLED:
                    executor.reset_section_counters()
                    part_text, tool_log = agentic_chat_completion(
                        client=client, model=text_model_id,
                        messages=messages, tools=tools,
                        tool_executor=executor.execute, temperature=config.WRITING_TEMPERATURE,
                        # Gleiche Parameter wie der non-tool Pfad für konsistente Ausgabe
                        frequency_penalty=0.5,
                        presence_penalty=0.4,
                        max_tokens=config.WRITING_MAX_TOKENS,
                    )
                    for tl in tool_log:
                        log(f"      🔧 Tool: {tl['name']}(...)")
                else:
                    part_text = robust_chat_completion(
                        client, text_model_id, messages,
                        temperature=0.1,
                        frequency_penalty=0.5,
                        presence_penalty=0.4,
                        max_tokens=config.WRITING_MAX_TOKENS,
                    )

                part_text = clean_llm_markdown_output(part_text)
                chapter_parts.append(part_text)
                last_section_full = part_text
                last_section_end = part_text[-400:].replace("\n", " ")
                
                if preview_callback:
                    temp_chapters = written_chapters + [f"# {ch_title}\n\n" + "\n\n".join(chapter_parts)]
                    inter_md = assemble_final_markdown(
                        parts=temp_chapters,
                        title=f"{final_book_title} (In Bearbeitung...)",
                        detail_level=detail_level,
                        source_files=[p.name for p in source_files],
                    )
                    preview_callback(inter_md)

            chapter_full = f"# {ch_title}\n\n" + "\n\n".join(chapter_parts)
            written_chapters.append(chapter_full)

        # ── PHASE 5: LOCAL LIBRARIAN (Bild-Placement) ─────────────────────────
        progress(84, "Bild-Placement...")
        log("🖼️ Bild-Placement (Embedding-Match)...")

        # Embedding-Modell erneut laden für Bild-Matching
        emb_model_id = switch_model(client, config.EMBEDDING_MODEL_ID, config.EMBEDDING_MODEL_ID)

        # Alle verfügbaren Bilder (mit Embedding) sammeln
        all_placeable_imgs = [
            img_data
            for c in containers
            for blk in c.topic_blocks
            for chk in blk.chunks
            for img_data in chk.attached_images
            if img_data.get("embedding") is not None
        ]

        def _run_placement_pass(threshold: float) -> tuple[list[str], set[str]]:
            """Führt einen Placement-Pass mit gegebenem Threshold durch."""
            placed: set[str] = set()
            chapters_out = list(written_chapters)
            for ch_idx, ch_text in enumerate(chapters_out):
                paragraphs = ch_text.split("\n\n")
                updated = []
                for para in paragraphs:
                    if not para.strip():
                        updated.append(para)
                        continue
                    para_emb = get_embedding(client, para[:500], emb_model_id)
                    if para_emb is None:
                        updated.append(para)
                        continue
                    best_score = threshold
                    best_img = None
                    for img_data in all_placeable_imgs:
                        if img_data["id"] in placed:
                            continue
                        sim = _cosine(para_emb, img_data["embedding"])
                        if sim > best_score:
                            best_score = sim
                            best_img = img_data
                    if best_img:
                        placed.add(best_img["id"])
                        img_rel = f"images/{Path(best_img['path']).name}"
                        updated.append(para)
                        desc = best_img.get("desc", "")[:120]
                        desc_html = markdown.markdown(desc)
                        if desc_html.startswith("<p>") and desc_html.endswith("</p>"):
                            desc_html = desc_html[3:-4]
                        updated.append(
                            f"![{best_img.get('desc', 'Abbildung')[:80]}]({img_rel})\n"
                            f"<p class='image-caption'>{desc_html}</p>"
                        )
                    else:
                        updated.append(para)
                chapters_out[ch_idx] = "\n\n".join(updated)
            return chapters_out, placed

        # Erster Placement-Pass mit konfiguriertem Threshold
        threshold = config.SYNTHESIS_IMAGE_PLACEMENT_THRESHOLD
        written_chapters, placed_imgs = _run_placement_pass(threshold)

        # Adaptiver Fallback: wenn >40% der Bilder unplatziert, Threshold senken
        if all_placeable_imgs:
            unplaced_ratio = 1 - len(placed_imgs) / len(all_placeable_imgs)
            if unplaced_ratio > 0.4 and threshold > 0.35:
                fallback_threshold = max(threshold - 0.05, 0.35)
                log(f"   🔄 Bild-Placement: {unplaced_ratio:.0%} unplatziert → Retry mit Threshold {fallback_threshold:.2f}")
                written_chapters, placed_imgs = _run_placement_pass(fallback_threshold)

        log(f"   🖼️ {len(placed_imgs)}/{len(all_placeable_imgs)} Bilder platziert")

        # RAG-Image-Tags auflösen
        written_chapters = [resolve_rag_image_tags(ch, images_dir) for ch in written_chapters]
        unload_model(emb_model_id)

        # ── PHASE 7: FINALIZATION ─────────────────────────────────────────────
        progress(90, "Dokument zusammenstellen...")
        log("📝 Finalisiere Dokument...")

        # Sicherer Dateiname aus Buchtitel — Umlaut-Transkription vor Sonderzeichen-Entfernung
        _st = final_book_title
        for _src, _dst in [("ä","ae"),("ö","oe"),("ü","ue"),("Ä","ae"),("Ö","oe"),("Ü","ue"),("ß","ss")]:
            _st = _st.replace(_src, _dst)
        safe_title = re.sub(r"[^\w\s\-]", "", _st).strip()
        safe_title = re.sub(r"\s+", "_", safe_title)[:80] or project_name

        final_md = assemble_final_markdown(
            parts=written_chapters,
            title=final_book_title,
            detail_level=detail_level,
            source_files=[p.name for p in source_files],
        )

        # Ausgabe-Ordner nach Buchtitel umbenennen
        named_output_dir = Path(config.CENTRAL_OUTPUT_DIR) / safe_title
        if not named_output_dir.exists():
            try:
                output_dir.rename(named_output_dir)
                output_dir = named_output_dir
                images_dir = output_dir / "images"
            except Exception:
                pass  # Fallback: ursprünglicher Ordner bleibt

        output_path = None
        if output_format in ("pdf", "both"):
            pdf_path = output_dir / f"{safe_title}.pdf"
            ok = markdown_to_pdf(final_md, pdf_path, output_dir, detail_level)
            if ok:
                output_path = pdf_path
                log(f"✅ PDF: {pdf_path}")
        if output_format in ("md", "both") or output_path is None:
            md_path = output_dir / f"{safe_title}.md"
            save_markdown(final_md, md_path)
            if output_path is None:
                output_path = md_path
            log(f"✅ Markdown: {md_path}")

        if evidence_pdf:
            log("🔎 Erstelle Evidence-PDF (Quellennachweis)...")
            evidence_md_parts = [
                f"# Quellen-Nachweis zu: {final_book_title}",
                "\n> Originaltexte zur Überprüfung.\n",
            ]
            for c in containers:
                evidence_md_parts.append(f"## Datei: {c.name}")
                for blk in c.topic_blocks:
                    evidence_md_parts.append(f"### {blk.title}")
                    for chk in blk.chunks:
                        if chk.text.strip():
                            evidence_md_parts.append(f"```text\n{chk.text.strip()}\n```\n")
            ev_pdf_path = output_dir / f"{safe_title}_Quellen.pdf"
            ok_ev = markdown_to_pdf(
                "\n".join(evidence_md_parts), ev_pdf_path, output_dir, detail_level
            )
            if ok_ev:
                log(f"✅ Evidence-PDF: {ev_pdf_path}")
            else:
                ev_md_path = output_dir / f"{safe_title}_Quellen.md"
                save_markdown("\n".join(evidence_md_parts), ev_md_path)

        if anki_export:
            _try_anki_export(written_chapters, output_dir, final_book_title, log)

        progress(100, "✅ Deep Synthesis abgeschlossen!")
        log(f"✅ Deep Synthesis abgeschlossen: {final_book_title}")
        return {"success": True, "output_path": output_path, "markdown": final_md, "error": ""}

    except InterruptedError as e:
        if log_callback:
            log_callback(f"⏹ {e}")
        return {"success": False, "output_path": None, "markdown": "", "error": str(e)}
    except Exception as e:
        import traceback
        err = f"{e}\n{traceback.format_exc()}"
        if log_callback:
            log_callback(f"❌❌❌ Kritischer Fehler: {e}")
        return {"success": False, "output_path": None, "markdown": "", "error": err}


# ── Anki-Export ───────────────────────────────────────────────────────────────

# Maximum characters of chapter text sent per LLM call. Longer chapters are
# split into overlapping windows so no content is missed.
_ANKI_CHUNK_SIZE = 3000
_ANKI_CHUNK_OVERLAP = 300


def _llm_generate_cards(text: str, section_hint: str, log) -> list[dict]:
    """
    Ask the LLM to generate 2-4 High-Yield exam Q&A pairs for *text*.

    Returns a list of dicts with keys "q" and "a", or an empty list on failure.
    The caller falls back to regex-based cards when this returns [].
    """
    try:
        from core.llm_client import create_openai_client, robust_chat_completion
        import config

        client = create_openai_client(
            host=config.LM_STUDIO_HOST,
            port=config.LM_STUDIO_PORT,
        )

        prompt = (
            f"Du bist ein Medizin-Dozent und erstellst Lernkarten für das Staatsexamen.\n\n"
            f"Thema: {section_hint}\n\n"
            f"Text:\n{text}\n\n"
            "Erstelle 2–4 essentielle Prüfungsfragen (High-Yield, Staatsexamen-Niveau) "
            "zu den wichtigsten Konzepten im obigen Text.\n"
            "Jede Frage soll präzise und klinisch relevant sein. "
            "Die Antwort soll vollständig, aber prägnant sein (2–5 Sätze).\n\n"
            "Antworte NUR als JSON-Liste ohne weiteren Text:\n"
            '[{"q": "Frage 1?", "a": "Antwort 1"}, {"q": "Frage 2?", "a": "Antwort 2"}]'
        )

        response = robust_chat_completion(
            client,
            config.TEXT_MODEL_LOAD,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        raw = response.choices[0].message.content or ""

        # Strip <think>…</think> blocks produced by reasoning models.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Extract the JSON array from the response (tolerates surrounding text).
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not json_match:
            return []

        import json
        cards = json.loads(json_match.group(0))

        # Keep only well-formed entries.
        return [c for c in cards if isinstance(c, dict) and "q" in c and "a" in c]

    except Exception as exc:
        log(f"⚠️ Anki LLM-Generierung fehlgeschlagen: {exc}")
        return []


def _regex_fallback_cards(text: str) -> list[dict]:
    """
    Fallback: extract **bold** terms from *text* and produce simple
    "Was ist X?" cards. Lower quality but always available.
    """
    bold_pattern = re.compile(r"\*\*(.+?)\*\*")
    cards = []
    for match in bold_pattern.finditer(text):
        term = match.group(1)
        start = max(0, match.start() - 120)
        ctx = text[start: match.end() + 120].replace("\n", " ").strip()
        cards.append({"q": f"Was ist: <b>{term}</b>?", "a": ctx})
    return cards


def _try_anki_export(chapters: list[str], output_dir: Path, name: str, log) -> None:
    """
    Generate an Anki deck (.apkg) from *chapters*.

    For each chapter the LLM is called to produce 2–4 Staatsexamen-level
    exam Q&A pairs. If a chapter is longer than _ANKI_CHUNK_SIZE characters
    it is split into overlapping windows. The bold-term regex is used as a
    fallback when the LLM call fails for a given chunk.
    """
    try:
        import genanki
        import random
    except ImportError:
        log("⚠️ genanki nicht installiert. Anki-Export übersprungen.")
        return

    try:
        model = genanki.Model(
            random.randrange(1 << 30, 1 << 31),
            "MedSkript",
            fields=[{"name": "Frage"}, {"name": "Antwort"}],
            templates=[{
                "name": "Card 1",
                "qfmt": "{{Frage}}",
                "afmt": "{{FrontSide}}<hr id=answer>{{Antwort}}",
            }],
        )
        deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), name)
        card_count = 0

        for idx, chapter in enumerate(chapters):
            section_hint = f"Abschnitt {idx + 1} von {len(chapters)}"

            # Split long chapters into overlapping windows.
            chunks: list[str] = []
            if len(chapter) <= _ANKI_CHUNK_SIZE:
                chunks = [chapter]
            else:
                pos = 0
                while pos < len(chapter):
                    chunks.append(chapter[pos: pos + _ANKI_CHUNK_SIZE])
                    pos += _ANKI_CHUNK_SIZE - _ANKI_CHUNK_OVERLAP

            for chunk in chunks:
                cards = _llm_generate_cards(chunk, section_hint, log)

                if not cards:
                    # LLM unavailable or returned unusable output — use regex.
                    cards = _regex_fallback_cards(chunk)

                for card in cards:
                    note = genanki.Note(
                        model=model,
                        fields=[str(card["q"]), str(card["a"])],
                    )
                    deck.add_note(note)
                    card_count += 1

        pkg = genanki.Package(deck)
        anki_path = output_dir / f"{name}.apkg"
        pkg.write_to_file(str(anki_path))
        log(f"🃏 Anki-Deck: {card_count} Karten → {anki_path}")

    except Exception as e:
        log(f"⚠️ Anki-Export Fehler: {e}")
