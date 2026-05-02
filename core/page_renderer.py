# core/page_renderer.py — PyMuPDF page rendering, Docling figure extraction, and analysis cache

# PyMuPDF page rendering, Docling figure extraction, and analysis cache.
# IMPORTANT: call render_*() and extract_figures_from_pdf() only in worker process.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF (fitz) nicht verfügbar. OCR-Modus deaktiviert.")
    print("   → Lösung: pip install pymupdf")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ── Data structures ───────────────────────────────────────────────────────────

# Represents an image extracted by Docling from a PDF page.
@dataclass
class DoclingFigure:
    """
    Represents an image extracted by Docling from a PDF page.
    All page numbers are 1-indexed (Docling convention).
    """
    figure_id: str          # e.g. "fig_008_0" (page_page_index)
    page_no: int            # 1-indexed (Docling convention)
    figure_index: int       # Index within page (0-based)
    image_path: Path        # Where the image is stored on disk
    description: str = ""   # Optional: Docling description or empty

    def to_dict(self) -> dict:
        return {
            "figure_id": self.figure_id,
            "page_no": self.page_no,
            "figure_index": self.figure_index,
            "image_path": str(self.image_path),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DoclingFigure":
        return cls(
            figure_id=d["figure_id"],
            page_no=d["page_no"],
            figure_index=d["figure_index"],
            image_path=Path(d["image_path"]),
            description=d.get("description", ""),
        )


# Group of consecutive pages with the same main topic.
@dataclass
class PageGroup:
    """
    Group of consecutive pages with the same main topic.
    Basic processing unit in OCR mode.
    All page numbers are 0-indexed (PyMuPDF convention).
    """
    file_path: Path
    pages: list[int]                          # 0-indexed page numbers
    main_topic: str
    sub_topic: str
    is_new_main_topic: bool = True
    figures: list[DoclingFigure] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": str(self.file_path),
            "pages": self.pages,
            "main_topic": self.main_topic,
            "sub_topic": self.sub_topic,
            "is_new_main_topic": self.is_new_main_topic,
            "figures": [f.to_dict() for f in self.figures],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageGroup":
        return cls(
            file_path=Path(d["file_path"]),
            pages=d["pages"],
            main_topic=d["main_topic"],
            sub_topic=d.get("sub_topic", ""),
            is_new_main_topic=d.get("is_new_main_topic", True),
            figures=[DoclingFigure.from_dict(f) for f in d.get("figures", [])],
        )


@dataclass
class OCRFileAnalysis:
    """Ergebnis des Analysis-Passes für eine einzelne PDF-Datei."""
    file_path: Path
    page_count: int
    page_groups: list[PageGroup]
    cache_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "file_path": str(self.file_path),
            "page_count": self.page_count,
            "page_groups": [g.to_dict() for g in self.page_groups],
        }

    @classmethod
    def from_dict(cls, d: dict, cache_path: Path | None = None) -> "OCRFileAnalysis":
        return cls(
            file_path=Path(d["file_path"]),
            page_count=d["page_count"],
            page_groups=[PageGroup.from_dict(g) for g in d.get("page_groups", [])],
            cache_path=cache_path,
        )


@dataclass
class OCRUnifiedTopic:
    """
    Für Deep Synthesis: Ein Topic das Seiten aus mehreren Dateien zusammenfasst.
    sub_sections: optionale Unterstruktur vom Sub-Architekten.
    Format: [{"sub_title": str, "group_indices_per_source": [[int,...], [int,...], ...]}]
    Wenn None → schreibt der Writing-Pass ohne semantische Unterstruktur.
    """
    main_topic: str
    sub_topic: str
    sources: list[dict]   # [{'file_path': Path, 'pages': [0,1,2]}, ...]
    figures: list[DoclingFigure] = field(default_factory=list)
    sub_sections: list[dict] | None = None  # Sub-Architekt-Ergebnis


# ── Seitenrendering ───────────────────────────────────────────────────────────

def get_page_count(pdf_path: Path) -> int:
    """
    Gibt die Seitenanzahl des PDFs zurück.
    Gibt 0 zurück wenn PyMuPDF nicht verfügbar oder Fehler.
    """
    if not PYMUPDF_AVAILABLE:
        return 0
    try:
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        print(f"   ⚠️ Seitenanzahl-Fehler ({pdf_path.name}): {e}")
        return 0


def render_pages(
    pdf_path: Path,
    page_nos: list[int],
    dpi: int,
) -> list["PILImage"]:
    """
    Rendert eine Liste von Seiten als PIL RGB Images.

    - page_nos: 0-indexed Seitennummern
    - dpi: Auflösung (96 für Analysis, 150 für Writing)
    - Öffnet PDF einmal, schließt es explizit
    - Gibt leere Liste bei Fehler oder fehlendem PyMuPDF zurück (nie werfen)
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        return []
    if not page_nos:
        return []
    try:
        doc = fitz.open(str(pdf_path))
        images = []
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for page_no in page_nos:
            if page_no < 0 or page_no >= len(doc):
                continue
            page = doc[page_no]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images
    except Exception as e:
        print(f"   ⚠️ render_pages Fehler ({pdf_path.name}): {e}")
        return []


def render_all_pages_as_thumbnails(
    pdf_path: Path,
    dpi: int | None = None,
) -> list["PILImage"]:
    """
    Rendert alle Seiten des PDFs als Thumbnails für den Analysis-Pass.
    Gibt leere Liste bei Fehler zurück (nie werfen).
    """
    if dpi is None:
        dpi = config.OCR_ANALYSIS_DPI
    count = get_page_count(pdf_path)
    if count == 0:
        return []
    return render_pages(pdf_path, list(range(count)), dpi)


# ── Cache-Utilities ───────────────────────────────────────────────────────────

def get_analysis_cache_key(pdf_path: Path) -> str:
    """
    Berechnet einen Cache-Schlüssel: sha256(erste 64KB der PDF) + DPI + Batch-Größe.
    Gibt leeren String bei Fehler zurück.
    """
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(65536)
        pdf_hash = hashlib.sha256(header).hexdigest()[:16]
        dpi = config.OCR_ANALYSIS_DPI
        batch = config.OCR_MAX_PAGES_PER_ANALYSIS_BATCH
        return f"{pdf_hash}_dpi{dpi}_batch{batch}"
    except Exception as e:
        print(f"   ⚠️ Cache-Key-Fehler: {e}")
        return ""


def load_analysis_cache(pdf_path: Path) -> OCRFileAnalysis | None:
    """
    Lädt gecachten Analysis-Pass aus OCR_ANALYSIS_CACHE_DIR.
    Gibt None zurück wenn kein Cache vorhanden oder ungültig.
    """
    if not config.OCR_ANALYSIS_CACHE_ENABLED:
        return None
    key = get_analysis_cache_key(pdf_path)
    if not key:
        return None
    cache_file = Path(config.OCR_ANALYSIS_CACHE_DIR) / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        analysis = OCRFileAnalysis.from_dict(data, cache_path=cache_file)
        # Figuren-Pfade validieren: mindestens 1 muss existieren
        all_figs_valid = all(
            fig.image_path.exists()
            for group in analysis.page_groups
            for fig in group.figures
        )
        if not all_figs_valid:
            print(f"   ⚠️ Cache-Bilder fehlen, ignoriere Cache für {pdf_path.name}")
            return None
        print(f"   ✅ Analysis-Cache geladen: {len(analysis.page_groups)} Gruppen")
        return analysis
    except Exception as e:
        print(f"   ⚠️ Cache-Lesen fehlgeschlagen: {e}")
        return None


def save_analysis_cache(analysis: OCRFileAnalysis) -> None:
    """
    Speichert OCRFileAnalysis als JSON in OCR_ANALYSIS_CACHE_DIR.
    Ignoriert Fehler.
    """
    if not config.OCR_ANALYSIS_CACHE_ENABLED:
        return
    key = get_analysis_cache_key(analysis.file_path)
    if not key:
        return
    try:
        cache_dir = Path(config.OCR_ANALYSIS_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
        analysis.cache_path = cache_file
    except Exception as e:
        print(f"   ⚠️ Cache-Speichern fehlgeschlagen: {e}")


# ── Docling Figuren-Extraktion ────────────────────────────────────────────────

def extract_figures_from_pdf(
    pdf_path: Path,
    figures_dir: Path,
    deduplicator=None,
) -> list[DoclingFigure]:
    """
    Extrahiert PictureItems aus PDF via Docling und speichert sie als JPEG.
    WICHTIG: Nur im Worker-Prozess aufrufen (Deadlock-Prävention).
    Gibt leere Liste bei Fehler zurück (nie werfen).
    """
    try:
        from docling_core.types.doc import PictureItem
    except ImportError:
        print("   ⚠️ docling_core nicht verfügbar — keine Figuren-Extraktion")
        return []

    from core.pdf_parser import build_docling_converter, ingest_pdf, is_image_in_margin
    from core.image_utils import save_pil_image, get_image_hash

    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
        converter = build_docling_converter()
        all_items, doc = ingest_pdf(pdf_path, converter)
    except Exception as e:
        print(f"   ⚠️ Docling-Figuren-Extraktion fehlgeschlagen: {e}")
        return []

    figures: list[DoclingFigure] = []
    page_figure_count: dict[int, int] = {}

    for item, _ in all_items:
        if not isinstance(item, PictureItem):
            continue
        try:
            page_no = item.prov[0].page_no  # 1-indexed

            # Margin-Check
            if is_image_in_margin(item, doc):
                continue

            # Bild aus Docling holen
            pil_img = item.get_image(doc)
            if pil_img is None:
                continue

            # Mindestgröße prüfen
            if pil_img.width < config.MIN_PIXEL_SIDE or pil_img.height < config.MIN_PIXEL_SIDE:
                continue

            # Dedup-Check
            if deduplicator is not None and deduplicator.is_duplicate(pil_img):
                continue

            # figure_id erzeugen
            fig_index = page_figure_count.get(page_no, 0)
            figure_id = f"fig_{page_no:03d}_{fig_index}"
            page_figure_count[page_no] = fig_index + 1

            # Speichern
            img_path = figures_dir / f"{figure_id}.jpg"
            success = save_pil_image(
                pil_img, img_path, quality=config.OCR_FIGURE_JPEG_QUALITY
            )
            if not success:
                continue

            figures.append(DoclingFigure(
                figure_id=figure_id,
                page_no=page_no,
                figure_index=fig_index,
                image_path=img_path,
                description="",
            ))

        except Exception as e:
            print(f"   ⚠️ Figur-Extraktion übersprungen: {e}")
            continue

    print(f"   📷 {len(figures)} Abbildungen aus Docling extrahiert")
    
    # Explizites Cleanup um Speicher freizugeben
    try:
        del all_items
        del doc
        del converter
        import gc
        gc.collect()
    except Exception:
        pass
        
    return figures


def get_figures_for_pages(
    figures: list[DoclingFigure],
    pages: list[int],    # 0-indexed!
    max_figures: int | None = None,
) -> list[DoclingFigure]:
    """
    Gibt Figuren zurück die zu den angegebenen Seiten gehören.
    Konvertiert 0-indexed pages → 1-indexed (Docling: page_no + 1).
    Begrenzt auf max_figures.
    """
    if max_figures is None:
        max_figures = config.OCR_MAX_FIGURES_PER_SECTION

    # 0-indexed → 1-indexed Set
    page_set_1indexed = {p + 1 for p in pages}

    result = [f for f in figures if f.page_no in page_set_1indexed]
    result.sort(key=lambda f: (f.page_no, f.figure_index))
    return result[:max_figures]


def build_figure_list_text(figures: list[DoclingFigure], figure_aliases: dict[str, str] | None = None) -> str:
    """
    Erzeugt einen menschenlesbaren Text der verfügbaren Figuren für den System-Prompt.
    Gibt leeren String zurück wenn keine Figuren vorhanden.
    figure_aliases: {figure_id: "Figur_1"} — wenn übergeben, werden Aliases angezeigt.
    """
    if not figures:
        return ""
    lines = [
        "\nVERFÜGBARE ABBILDUNGEN FÜR DIESEN ABSCHNITT:",
    ]
    for fig in figures:
        alias = figure_aliases.get(fig.figure_id, fig.figure_id) if figure_aliases else fig.figure_id
        desc = f" — {fig.description}" if fig.description else ""
        lines.append(f"  - {alias}: Seite {fig.page_no}{desc}")
    lines.append(
        "\nNutze insert_figure(figure_id=\"Figur_1\", caption=\"...\") um eine Abbildung "
        "einzufügen. Nur Alias-Namen aus der obigen Liste sind gültig. Max. 3 Abbildungen pro Abschnitt."
    )
    return "\n".join(lines)


# ── Writing Cache ─────────────────────────────────────────────────────────────

def get_writing_cache_key(
    main_topic: str,
    sub_topic: str,
    page_info: list, # [(fname, pno), ...]
    model_id: str,
    detail_level: int,
) -> str:
    """Erzeugt einen stabilen MD5-Hash für einen Writing-Batch."""
    import hashlib
    import json
    # Relevante Parameter serialisieren
    data = {
        "m": main_topic,
        "s": sub_topic,
        "p": sorted(page_info), # Sortiert für Stabilität
        "mod": model_id,
        "det": detail_level,
    }
    s = json.dumps(data, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_writing_cache(cache_key: str) -> str | None:
    """Lädt einen Abschnittstext aus dem Cache wenn vorhanden."""
    cache_path = config.OCR_WRITING_CACHE_DIR / f"{cache_key}.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

def save_writing_cache(cache_key: str, text: str) -> None:
    """Speichert einen fertigen Abschnittstext im Cache."""
    try:
        config.OCR_WRITING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = config.OCR_WRITING_CACHE_DIR / f"{cache_key}.txt"
        cache_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"   ⚠️ Writing-Cache Speichern fehlgeschlagen: {e}")


# ── Text-Mode Writing Cache ───────────────────────────────────────────────────
# Same resume-on-abort semantics as the OCR writing cache, but stored under
# TEXT_WRITING_CACHE_DIR and keyed on section content rather than page numbers.

def get_text_writing_cache_key(content: str, model_id: str, detail_level: int) -> str:
    """
    Stable MD5 hash for a text-mode writing section.

    content      – raw source text (chunk or full_txt) fed to the LLM.
    model_id     – separate cache entry when model changes.
    detail_level – separate entry for different output styles.
    """
    content_hash = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
    data = {"c": content_hash, "mod": model_id, "det": detail_level}
    s = json.dumps(data, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_text_writing_cache(cache_key: str) -> str | None:
    """Load a cached text-mode section. Returns None when not found."""
    cache_path = config.TEXT_WRITING_CACHE_DIR / f"{cache_key}.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def save_text_writing_cache(cache_key: str, text: str) -> None:
    """Persist a finished text-mode section immediately after the LLM call."""
    try:
        config.TEXT_WRITING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = config.TEXT_WRITING_CACHE_DIR / f"{cache_key}.txt"
        cache_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"   ⚠️ Text-Writing-Cache Speichern fehlgeschlagen: {e}")
