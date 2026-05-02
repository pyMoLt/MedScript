# modes/__init__.py
from modes.summary import process_single_file
from modes.synthesis import process_deep_synthesis
from modes.ocr_summary import process_single_file_ocr
from modes.ocr_synthesis import process_multiple_files_ocr

__all__ = [
    "process_single_file", "process_deep_synthesis",
    "process_single_file_ocr", "process_multiple_files_ocr",
]
