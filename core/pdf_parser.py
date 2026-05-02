# core/pdf_parser.py — Docling wrapper, PDF repair, and ingestion pipeline

# Docling wrapper, PDF repair, ingestion pipeline.
# IMPORTANT: call build_docling_converter() and ingest_pdf() only in worker process.

from pathlib import Path
import logging

# Docling's underlying OCR (RapidOCR) is very verbose for empty regions
logging.getLogger("RapidOCR").setLevel(logging.ERROR)

import config

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# Attempts to repair PDF via pypdf (rewrite pages).
def repair_pdf_if_needed(source_path: Path) -> Path:
    """
    Attempts to repair PDF via pypdf (rewrite pages).
    Returns repaired_{name}.pdf or original on error/missing pypdf.
    Caller is responsible for cleaning up repaired_*.pdf file.
    """
    if not PYPDF_AVAILABLE:
        return source_path
    try:
        print(f"   🔧 Präventive PDF-Reparatur: {source_path.name}...", flush=True)
        reader = pypdf.PdfReader(str(source_path))
        writer = pypdf.PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        repaired = source_path.parent / f"repaired_{source_path.name}"
        with open(repaired, "wb") as f:
            writer.write(f)
        return repaired
    except Exception as e:
        print(f"   ⚠️ Reparatur fehlgeschlagen ({e}), nutze Original.", flush=True)
        return source_path


# Creates a configured DocumentConverter.
def build_docling_converter():
    """
    Creates a configured DocumentConverter.
    ONLY call in worker process (deadlock prevention).
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = False   # wir machen das selbst
    pipeline_options.do_table_structure = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# Converts PDF with Docling.
def ingest_pdf(source_path: Path, converter) -> tuple[list, object]:
    """
    Converts PDF with Docling.
    Returns (all_items, doc).
    all_items = list(doc.iterate_items())
    """
    result = converter.convert(str(source_path))
    doc = result.document
    all_items = list(doc.iterate_items())
    return all_items, doc


# Extracts text context around an element (for image analysis prompts).
def get_item_context(
    all_items: list,
    index: int,
    lookback: int = 4,
    lookahead: int = 4,
    max_chars: int = 200,
) -> str:
    """
    Extracts text context around an element (for image analysis prompts).
    """
    # Lazy import only when needed (worker process)
    try:
        from docling_core.types.doc import TextItem
    except ImportError:
        return ""

    before_parts = []
    for k in range(1, lookback + 1):
        i = index - k
        if i < 0:
            break
        item, _ = all_items[i]
        if isinstance(item, TextItem):
            before_parts.insert(0, item.text[:max_chars])

    after_parts = []
    for k in range(1, lookahead + 1):
        i = index + k
        if i >= len(all_items):
            break
        item, _ = all_items[i]
        if isinstance(item, TextItem):
            after_parts.append(item.text[:max_chars])

    before = "\n".join(before_parts)
    after = "\n".join(after_parts)
    return f"{before}\n--- BILD ---\n{after}".strip()


# Checks if an image is in the header/footer area of the page (default: top/bottom 8%).
def is_image_in_margin(item, doc, margin: float = 0.08) -> bool:
    """
    Checks if an image is in the header/footer area of the page (default: top/bottom 8%).
    Returns True if in margin (should be skipped).
    Returns False on error (safe default: keep image).
    """
    try:
        bbox = item.prov[0].bbox
        page_h = doc.pages[item.prov[0].page_no].size.height
        mid_y = (bbox.b + bbox.t) / 2
        return mid_y > (page_h * (1.0 - margin)) or mid_y < (page_h * margin)
    except Exception:
        return False
