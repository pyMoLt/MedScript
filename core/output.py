# core/output.py — Markdown to HTML to PDF conversion via WeasyPrint with CSS

import re
from pathlib import Path

import config

try:
    import markdown as md_lib
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from weasyprint import HTML as WPHtml
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False
    print("⚠️ WeasyPrint nicht verfügbar. PDF-Ausgabe deaktiviert (nur Markdown).")
    print("   → Lösung: brew install pango gobject-introspection glib")


# ── CSS definitions ───────────────────────────────────────────────────────────────

DEFAULT_CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center {
        content: "Seite " counter(page);
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-size: 9pt;
        color: #888;
    }
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #2c3e50;
    text-align: justify;
    hyphens: auto;
    orphans: 3;
    widows: 3;
}
a { color: inherit; text-decoration: none; }
.cover-page {
    height: 100vh; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
    background: radial-gradient(circle at center, #fdfbf7 0%, #ebedee 100%);
    border: 20px solid #fff; box-sizing: border-box; padding: 60px;
    page-break-after: always;
}
.cover-title {
    font-size: 38pt; font-weight: 800; color: #2C3E50;
    margin-bottom: 30px; letter-spacing: -1.5px; line-height: 1.1;
    text-transform: uppercase;
}
.cover-subtitle {
    font-size: 14pt; font-weight: 300; color: #16a085;
    margin-bottom: 60px; letter-spacing: 3px;
    border-top: 1px solid #16a085; border-bottom: 1px solid #16a085;
    padding: 10px 40px; display: inline-block;
}
h1 {
    font-size: 22pt; color: #2C3E50; margin-top: 40px; margin-bottom: 20px;
    font-weight: 800; border-bottom: 3px solid #ecf0f1; padding-bottom: 10px;
    page-break-before: always; letter-spacing: -0.5px;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
    font-size: 15pt; color: #16a085; margin-top: 28px; margin-bottom: 10px;
    font-weight: 700; page-break-after: avoid;
}
h3 {
    font-size: 12pt; color: #555; text-transform: uppercase;
    letter-spacing: 1px; margin-top: 22px;
    border-left: 3px solid #ccc; padding-left: 10px; page-break-after: avoid;
}
p { margin-bottom: 10px; }
strong { color: #000; font-weight: 700; }
ul, ol { margin-bottom: 14px; padding-left: 22px; }
li { margin-bottom: 5px; }
img {
    width: auto; max-width: 100%; max-height: 42vh; height: auto;
    display: block; margin: 20px auto 8px auto;
    border-radius: 2px; border: 1px solid #ddd; padding: 4px;
    background: #fff; box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    page-break-inside: avoid;
}
.image-caption {
    display: block; text-align: center; font-size: 0.88em;
    color: #666; margin-bottom: 28px; font-style: italic;
    page-break-before: avoid;
}
.toc-container {
    column-count: 2;
    column-gap: 1.5cm;
    margin-bottom: 2cm;
    page-break-after: always;
}
.toc-container h2 {
    column-span: all;
    margin-top: 0;
    page-break-before: avoid;
}
.toc-container ul {
    list-style: none;
    padding-left: 0;
}
.toc-container li {
    margin-bottom: 4px;
    font-size: 10pt;
    break-inside: avoid;
}
.toc-container li a {
    display: flex;
    align-items: baseline;
}
.toc-container li a::after {
    content: leader(dotted) target-counter(attr(href), page);
    margin-left: 5px;
    color: #7f8c8d;
}
.toc-container li ul {
    margin-top: 2px;
    margin-left: 15px;
}
.toc-container li ul li a::after {
    font-size: 0.9em;
}
.analysis-box {
    background-color: #f4fdfb; border-left: 3px solid #1abc9c;
    border-radius: 2px; padding: 12px 18px; margin-top: 5px;
    margin-bottom: 22px; font-size: 0.9em; color: #2c3e50;
    page-break-inside: avoid;
}
.source-box {
    font-size: 0.75em; color: #7f8c8d; background-color: #fcfcfc;
    border-top: 1px solid #eee; padding: 8px 0; margin: 14px 0;
    font-style: italic; page-break-inside: avoid;
}
.source-box::before { content: "📚 "; }
blockquote {
    border-left: 4px solid #e67e22; margin: 22px 0; padding: 14px 20px;
    color: #d35400; background-color: #fffaf5;
    border-radius: 0 4px 4px 0; page-break-inside: avoid; font-weight: 500;
}
blockquote p { margin: 0; }
table {
    width: 100%; border-collapse: collapse; margin: 22px 0;
    font-size: 10pt; border: 1px solid #ddd;
}
th, td { border: 1px solid #ddd; padding: 9px 12px; text-align: left; vertical-align: top; }
th { background-color: #ecf0f1; color: #2c3e50; font-weight: 700; border-bottom: 2px solid #bdc3c7; }
tr:nth-child(even) { background-color: #fdfdfd; }
.ref-tag { font-size: 0.6em; color: #95a5a6; vertical-align: super; margin-left: 2px; font-family: monospace; }
"""

COMPACT_CSS = """
@page { size: A4; margin: 2cm 1.8cm; }
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10pt; line-height: 1.5; color: #333; text-align: left;
}
h1 { font-size: 18pt; color: #2C3E50; border-bottom: 2px solid #ccc;
     padding-bottom: 6px; margin-top: 20px; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 13pt; color: #16a085; margin-top: 16px; font-weight: 700; }
h3 { font-size: 11pt; color: #555; font-weight: bold; margin-top: 12px; }
p { margin-bottom: 6px; }
ul, ol { margin-bottom: 8px; padding-left: 18px; }
li { margin-bottom: 3px; }
img { max-width: 70%; max-height: 30vh; display: block; margin: 12px auto; border: 1px solid #ddd; }
blockquote { border-left: 3px solid #e67e22; margin: 10px 0; padding: 6px 14px; color: #d35400; background: #fffaf5; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 9pt; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background-color: #f0f0f0; font-weight: bold; }
"""


def get_css_for_detail_level(detail_level: int) -> str:
    return COMPACT_CSS if detail_level < 40 else DEFAULT_CSS


# ── PDF-Erstellung ────────────────────────────────────────────────────────────

# Converts Markdown to HTML to PDF via WeasyPrint.
def markdown_to_pdf(
    markdown_text: str,
    output_pdf_path: Path,
    base_url: Path,
    detail_level: int = 100,
) -> bool:
    """
    Converts Markdown to HTML to PDF via WeasyPrint.
    Returns True on success, False on error or missing WeasyPrint.
    """
    if not WEASYPRINT_AVAILABLE:
        print("⚠️ PDF-Erstellung übersprungen (WeasyPrint fehlt).")
        return False
    if not MARKDOWN_AVAILABLE:
        print("⚠️ 'markdown' Paket fehlt.")
        return False
    try:
        css = get_css_for_detail_level(detail_level)
        # Use a custom TocExtension so heading id= attributes use the same
        # slugify function as _generate_toc(). This prevents broken TOC links
        # when headings contain German umlauts — the default toc extension
        # maps ü→u while _slugify maps ü→ue, causing href/id mismatches.
        try:
            from markdown.extensions.toc import TocExtension
            toc_ext = TocExtension(slugify=_slugify_toc)
        except Exception:
            toc_ext = "toc"  # graceful fallback to string form
        html_body = md_lib.markdown(
            markdown_text,
            extensions=["extra", "admonition", "nl2br", "sane_lists", toc_ext],
        )
        full_html = (
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            f"<style>{css}</style>"
            f"</head><body>{html_body}</body></html>"
        )
        WPHtml(string=full_html, base_url=str(base_url)).write_pdf(
            target=str(output_pdf_path)
        )
        return True
    except Exception as e:
        print(f"❌ PDF-Render-Fehler: {e}")
        return False


# Saves Markdown as .md file. Returns True on success.
def save_markdown(markdown_text: str, output_path: Path) -> bool:
    """Saves Markdown as .md file. Returns True on success."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        return True
    except Exception as e:
        print(f"❌ Markdown-Speicher-Fehler: {e}")
        return False


# Creates a URL-friendly anchor slug from a title.
def _slugify(text: str) -> str:
    """Creates a URL-friendly anchor slug from a title.
    Umlauts are transcribed before removing special characters (ä→ae, ö→oe, ü→ue, ß→ss),
    so German headings get readable anchors."""
    s = text.strip()
    # Umlaut-Transkription (Groß- und Kleinschreibung)
    for src, dst in [("ä", "ae"), ("ö", "oe"), ("ü", "ue"),
                     ("Ä", "ae"), ("Ö", "oe"), ("Ü", "ue"), ("ß", "ss")]:
        s = s.replace(src, dst)
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"[\s\-]+", "-", s)
    return s


def _slugify_toc(value: str, separator: str) -> str:
    """
    Adapter wrapping _slugify() for use as the TocExtension slugify parameter.
    The toc extension calls slugify(value, separator) — this bridges the
    signature difference so heading id= attributes always match the hrefs
    generated by _generate_toc().
    """
    result = _slugify(value)
    if separator != "-":
        result = result.replace("-", separator)
    return result


# Generates a clickable table of contents from ## and ### headers in parts list.
def _generate_toc(parts: list[str]) -> str:
    """
    Generates a clickable table of contents from ## and ### headers in parts list.
    """
    lines = []
    for part in parts:
        for line in part.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## "):
                title = stripped[3:].strip().lstrip("#").strip()
                if title:
                    slug = _slugify(title)
                    lines.append(f"- [{title}](#{slug})")
            elif stripped.startswith("### "):
                title = stripped[4:].strip().lstrip("#").strip()
                if title:
                    slug = _slugify(title)
                    lines.append(f"  - [{title}](#{slug})")
    return "\n".join(lines) if lines else "_Kein Inhaltsverzeichnis verfügbar._"


# Assembles all text parts into a complete Markdown document.
def assemble_final_markdown(
    parts: list[str],
    title: str,
    detail_level: int,
    source_files: list[str] = None,
) -> str:
    """
    Assembles all text parts into a complete Markdown document.
    Table of contents is generated programmatically from headers in parts
    (renderer-independent, works in .md viewers and WeasyPrint).
    """
    toc_md = _generate_toc(parts)
    # Pre-render the TOC markdown to HTML before embedding it inside the div.
    # Python-Markdown does NOT process markdown inside block-level HTML elements
    # unless the element carries markdown="1" — pre-rendering avoids raw link
    # text appearing literally in the PDF.
    if MARKDOWN_AVAILABLE:
        toc_html = md_lib.markdown(toc_md, extensions=["sane_lists"])
    else:
        toc_html = f"<p>{toc_md}</p>"
    header_lines = [
        f"# {title}",
        f"> Generiertes Lernskript | Detailgrad: {detail_level}%",
        "",
        '<div class="toc-container">',
        "<h2>Inhaltsverzeichnis</h2>",
        toc_html,
        "</div>",
        "",
    ]
    if source_files:
        header_lines += [
            "**Quellen:**",
            *[f"- {f}" for f in source_files],
            "",
        ]

    full = "\n".join(header_lines) + "\n\n" + "\n\n".join(p for p in parts if p.strip())
    # Max 2 consecutive blank lines
    full = re.sub(r'\n{3,}', '\n\n', full)
    return full
