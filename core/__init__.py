# core/__init__.py
# Re-exports der am häufigsten genutzten Symbole.

from core.llm_client import (
    create_openai_client,
    robust_chat_completion,
    agentic_chat_completion,
    get_embedding,
    switch_model,
    unload_model,
    is_server_reachable,
    ensure_lm_studio_running,
    extract_json_robust,
)
from core.text_utils import (
    is_text_junk,
    smart_split_text,
    clean_llm_markdown_output,
    chunk_text_with_overlap,
    inject_markdown_headers,
)
from core.image_utils import (
    pil_to_base64_jpeg,
    save_pil_image,
    get_image_hash,
    AdvancedDeduplicator,
)
from core.pdf_parser import (
    repair_pdf_if_needed,
    build_docling_converter,
    ingest_pdf,
    get_item_context,
    is_image_in_margin,
)
from core.output import (
    markdown_to_pdf,
    save_markdown,
    assemble_final_markdown,
    DEFAULT_CSS,
    COMPACT_CSS,
)
from core.cache import SimpleCache
from core.settings_manager import get_settings
from core.page_renderer import (
    get_page_count,
    render_pages,
    render_all_pages_as_thumbnails,
    extract_figures_from_pdf,
    get_figures_for_pages,
    build_figure_list_text,
    load_analysis_cache,
    save_analysis_cache,
    get_analysis_cache_key,
    DoclingFigure,
    PageGroup,
    OCRFileAnalysis,
    OCRUnifiedTopic,
    PYMUPDF_AVAILABLE,
)

__all__ = [
    "create_openai_client", "robust_chat_completion", "agentic_chat_completion",
    "get_embedding", "switch_model", "unload_model", "is_server_reachable",
    "ensure_lm_studio_running", "extract_json_robust",
    "is_text_junk", "smart_split_text", "clean_llm_markdown_output",
    "chunk_text_with_overlap", "inject_markdown_headers",
    "pil_to_base64_jpeg", "save_pil_image", "get_image_hash", "AdvancedDeduplicator",
    "repair_pdf_if_needed", "build_docling_converter", "ingest_pdf",
    "get_item_context", "is_image_in_margin",
    "markdown_to_pdf", "save_markdown", "assemble_final_markdown",
    "DEFAULT_CSS", "COMPACT_CSS",
    "SimpleCache", "get_settings",
    # OCR-Modus
    "get_page_count", "render_pages", "render_all_pages_as_thumbnails",
    "extract_figures_from_pdf", "get_figures_for_pages", "build_figure_list_text",
    "load_analysis_cache", "save_analysis_cache", "get_analysis_cache_key",
    "DoclingFigure", "PageGroup", "OCRFileAnalysis", "OCRUnifiedTopic",
    "PYMUPDF_AVAILABLE",
]
