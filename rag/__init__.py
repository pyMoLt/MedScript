# rag/__init__.py
from rag.store import create_store, list_available_stores
from rag.retriever import retrieve, retrieve_for_section, batch_retrieve
from rag.tools import get_tools_for_run, ToolExecutor
from rag.augmenter import get_writer_tool_instruction, resolve_rag_image_tags
from rag.ocr_tools import OCRToolExecutor, INSERT_FIGURE_TOOL

__all__ = [
    "create_store", "list_available_stores",
    "retrieve", "retrieve_for_section", "batch_retrieve",
    "get_tools_for_run", "ToolExecutor",
    "get_writer_tool_instruction", "resolve_rag_image_tags",
    "OCRToolExecutor", "INSERT_FIGURE_TOOL",
]
