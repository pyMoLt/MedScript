# rag/indexer.py — Builds RAG store from textbook PDFs (text + images)

# Builds RAG store from textbook PDFs (text + images).
# Can be used as CLI: python main.py --index --input /path/to/books --store-name anatomie

from __future__ import annotations
import math
from pathlib import Path

import config
from core.llm_client import (
    create_openai_client, get_embedding, switch_model, unload_model,
    robust_chat_completion,
)
from core.pdf_parser import (
    repair_pdf_if_needed, build_docling_converter, ingest_pdf,
    get_item_context, is_image_in_margin,
)
from core.image_utils import AdvancedDeduplicator, save_pil_image, pil_to_base64_jpeg
from core.text_utils import is_text_junk, chunk_text_with_overlap
from rag.store import create_store


def build_rag_store_from_pdfs(
    input_paths: list[Path],
    store_name: str,
    force_rebuild: bool = False,
    append_mode: bool = False,
    index_images: bool = True,
    progress_callback: callable = None,
    log_callback: callable = None,
) -> dict:
    """
    Hauptfunktion. Indexiert alle PDFs in input_paths.
    Gibt zurück: {'success': bool, 'chunks_added': int, 'error': str}
    """

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    try:
        # ── SCHRITT 1: Vorbereitung ───────────────────────────────────────────
        log(f"📚 RAG-Indexierung startet: {len(input_paths)} Datei(en) → Store '{store_name}'")
        
        store_path = config.RAG_STORES_DIR / store_name
        chroma_path = store_path / "chroma"
        
        # Prüfen ob Store existiert (via Filesystem, nicht via DB-Handle um Locks zu vermeiden)
        store_exists = chroma_path.exists()
        
        if store_exists and not force_rebuild and not append_mode:
            # Hier müssen wir kurz den Store öffnen um die Anzahl zu zählen (nur für die Log-Meldung)
            temp_store = create_store(store_name)
            count = temp_store.count()
            log(f"✅ Store '{store_name}' existiert bereits ({count} Chunks). Abbruch, da weder 'Erweitern' noch 'Überschreiben' gewählt wurde.")
            return {"success": True, "chunks_added": 0, "error": "Store existiert bereits."}
            
        if store_exists and append_mode and not force_rebuild:
            temp_store = create_store(store_name)
            log(f"➕ Store '{store_name}' existiert bereits ({temp_store.count()} Chunks). Erweitere Store...")
            store = temp_store
        elif force_rebuild:
            if store_exists:
                log(f"🗑️ Lösche bestehenden Store '{store_name}' (Dateisystem)...")
                import shutil
                # Wir löschen den kompletten Ordner direkt auf Datesebene, 
                # OHNE vorher eine DB-Verbindung (create_store) zu öffnen.
                try:
                    shutil.rmtree(store_path, ignore_errors=True)
                except Exception as e:
                    log(f"⚠️ Warnung beim Löschen: {e}")
            
            log(f"🆕 Erstelle neuen Store '{store_name}'...")
            store = create_store(store_name)
        else:
            # Ersterstellung
            store = create_store(store_name)

        client = create_openai_client()

        # Bild-Speicherverzeichnis
        images_dir = config.RAG_STORES_DIR / store_name / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        all_text_chunks: list[dict] = []
        all_image_chunks: list[dict] = []
        global_dedup = AdvancedDeduplicator()

        n_files = len(input_paths)

        # ── SCHRITT 2 + 3: Text- und Bild-Extraktion (für jede PDF) ──────────
        for file_idx, pdf_path in enumerate(input_paths):
            base_pct = int(file_idx / n_files * 40)
            log(f"📄 Verarbeite: {pdf_path.name} ({file_idx + 1}/{n_files})")
            progress(base_pct, f"Lese {pdf_path.name}...")

            # PDF reparieren falls nötig
            working_path = repair_pdf_if_needed(pdf_path)

            try:
                converter = build_docling_converter()
                all_items, doc = ingest_pdf(working_path, converter)
            except Exception as e:
                log(f"❌❌❌ Docling-Fehler bei {pdf_path.name}: {e}")
                continue
            finally:
                if working_path != pdf_path and working_path.exists():
                    try:
                        working_path.unlink()
                    except Exception:
                        pass

            stem = pdf_path.stem.replace(" ", "_")[:40]

            # Text-Extraktion
            full_text_parts = []
            page_approx = 1
            try:
                from docling_core.types.doc import TextItem, TableItem, PictureItem
            except ImportError:
                from docling.datamodel.base_models import TextItem, TableItem, PictureItem  # type: ignore

            img_idx = 0
            for item_idx, (item, level) in enumerate(all_items):
                # Text und Tabellen
                if isinstance(item, (TextItem, TableItem)):
                    try:
                        text = item.text.strip()
                        label = str(getattr(item, "label", "")).lower()
                    except Exception:
                        continue
                    if is_text_junk(text, label):
                        continue
                    full_text_parts.append(text)
                    # Grobe Seitenzahl abschätzen
                    if hasattr(item, "prov") and item.prov:
                        try:
                            page_approx = item.prov[0].page_no
                        except Exception:
                            pass

                # Bilder — nur wenn index_images aktiv
                elif isinstance(item, PictureItem) and index_images:
                    try:
                        pil_img = item.get_image(doc)
                        if pil_img is None:
                            continue
                        w, h = pil_img.size
                        if min(w, h) < config.MIN_PIXEL_SIDE:
                            continue
                        if is_image_in_margin(item, doc):
                            continue
                        if global_dedup.is_duplicate(pil_img, f"{stem}_{img_idx}"):
                            continue
                        # Kontext extrahieren
                        ctx = get_item_context(all_items, item_idx)
                        # Bild speichern
                        img_filename = f"{stem}__img{img_idx}.jpg"
                        img_save_path = images_dir / img_filename
                        save_pil_image(pil_img, img_save_path)
                        img_idx += 1

                        # Bild-Beschreibung (Vision-Modell, wird in Schritt nach Text-Embedding aufgerufen)
                        all_image_chunks.append({
                            "_pil": pil_img,
                            "_ctx": ctx,
                            "_path": str(img_save_path),
                            "_source": pdf_path.name,
                            "_page": page_approx,
                            "_stem": stem,
                            "_img_idx": img_idx - 1,
                        })
                    except Exception as e:
                        log(f"   ⚠️ Bild-Extraktion Fehler: {e}")

            # Text in RAG-Chunks aufteilen
            full_text = " ".join(full_text_parts)
            raw_chunks = chunk_text_with_overlap(full_text, config.RAG_CHUNK_SIZE, config.RAG_CHUNK_OVERLAP)
            for c_idx, ch in enumerate(raw_chunks):
                all_text_chunks.append({
                    "id": f"{stem}__p{page_approx}__c{c_idx}",
                    "text": ch["text"],
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_approx,
                        "type": "text",
                        "image_path": "",
                    },
                })

        log(f"✅ Extraktion abgeschlossen: {len(all_text_chunks)} Text-Chunks, {len(all_image_chunks)} Bilder")

        # ── Bild-Analysen mit Vision-Modell ───────────────────────────────────
        final_image_chunks = []
        if all_image_chunks:
            log(f"👁️ Starte Bild-Analyse ({len(all_image_chunks)} Bilder)...")
            progress(45, "Vision-Modell lädt...")
            model_id = switch_model(client, config.VISION_MODEL_SEARCH, config.VISION_MODEL_LOAD)

            for i, img_data in enumerate(all_image_chunks):
                progress(45 + int(i / len(all_image_chunks) * 20), f"Bild {i + 1}/{len(all_image_chunks)} analysiert...")
                try:
                    img_b64 = pil_to_base64_jpeg(img_data["_pil"])
                    ctx = img_data["_ctx"]
                    prompt = (
                        f"Analysiere dieses Bild aus einem medizinischen Lehrbuch für eine Wissensdatenbank.\n"
                        f"Kontext (umgebender Text):\n\"\"\"{ctx}\"\"\"\n\n"
                        f"Erstelle eine DICHTE, VOLLSTÄNDIGE Beschreibung auf Deutsch.\n"
                        f"Beschreibe:\n"
                        f"1. Was ist zu sehen? (Struktur, Typ: Histologie/Schema/Foto/Diagramm/Tabelle)\n"
                        f"2. Welche anatomischen/physiologischen Strukturen oder Konzepte?\n"
                        f"3. Welche Beschriftungen/Labels sind vorhanden? (Bitte transkribieren)\n"
                        f"4. Welche Mechanismen oder Abläufe werden dargestellt?\n"
                        f"5. Was ist die Kernaussage?\n"
                        f"Format: Strukturierter Fließtext mit **Fettungen** für Schlüsselbegriffe.\n"
                        f"Maximal {config.RAG_IMAGE_DESC_TOKENS} Tokens."
                    )
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_b64}},
                        ]},
                    ]
                    desc = robust_chat_completion(client, model_id, messages)
                    if desc.startswith("[FEHLER"):
                        log(f"   ⚠️ Bild-Analyse fehlgeschlagen: {desc}")
                        continue

                    stem = img_data["_stem"]
                    img_idx_val = img_data["_img_idx"]
                    final_image_chunks.append({
                        "id": f"{stem}__img{img_idx_val}",
                        "text": f"[LEHRBUCH-BILD] Quelle: {img_data['_source']}\n{desc}",
                        "metadata": {
                            "source": img_data["_source"],
                            "page": img_data["_page"],
                            "type": "image_description",
                            "image_path": img_data["_path"],
                        },
                    })
                except Exception as e:
                    log(f"   ❌ Bild-Analyse Fehler: {e}")

        # ── SCHRITT 4: Embedding ──────────────────────────────────────────────
        progress(65, "Embedding-Modell lädt...")
        log(f"🔢 Berechne Embeddings für {len(all_text_chunks) + len(final_image_chunks)} Chunks...")
        emb_model_id = switch_model(client, config.EMBEDDING_MODEL_ID, config.EMBEDDING_MODEL_ID)

        all_chunks_to_store = all_text_chunks + final_image_chunks
        n_total = len(all_chunks_to_store)
        embedded_chunks = []
        for i, chunk in enumerate(all_chunks_to_store):
            progress(65 + int(i / n_total * 25), f"Embedding {i + 1}/{n_total}...")
            emb = get_embedding(client, chunk["text"], emb_model_id)
            if emb is None:
                log(f"   ⚠️ Embedding fehlgeschlagen für Chunk {chunk['id']} — übersprungen")
                continue
            chunk["embedding"] = emb
            embedded_chunks.append(chunk)

        # ── SCHRITT 5: Store befüllen ─────────────────────────────────────────
        progress(90, "Schreibe in Store...")
        log(f"💾 Schreibe {len(embedded_chunks)} Chunks in Store '{store_name}'...")
        store.add_chunks(embedded_chunks)

        # ── SCHRITT 6: Modell entladen ────────────────────────────────────────
        unload_model(emb_model_id)
        progress(100, "Indexierung abgeschlossen!")
        log(f"✅ Indexierung abgeschlossen: {len(embedded_chunks)} Chunks in Store '{store_name}'")
        return {"success": True, "chunks_added": len(embedded_chunks), "error": ""}

    except Exception as e:
        msg = f"Indexierung fehlgeschlagen: {e}"
        log(f"❌❌❌ {msg}")
        return {"success": False, "chunks_added": 0, "error": msg}


def estimate_indexing_time(input_paths: list[Path]) -> str:
    """Grobe Zeitschätzung für den Indexierungsvorgang."""
    # Schätzung: ~30 Seiten/Minute Text, ~2 Min/Bild Vision-Analyse
    total_pages = 0
    for p in input_paths:
        try:
            size_mb = p.stat().st_size / 1_000_000
            total_pages += max(1, int(size_mb * 20))  # Grobe Schätzung: 20 Seiten/MB
        except Exception:
            total_pages += 50
    text_minutes = max(1, total_pages // 30)
    image_minutes = max(0, (total_pages // 5) * 2)  # 1 Bild alle 5 Seiten, 2 Min/Bild
    total_minutes = text_minutes + image_minutes
    if total_minutes < 60:
        return f"ca. {total_minutes} Minuten"
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"ca. {hours}h {mins}min"


