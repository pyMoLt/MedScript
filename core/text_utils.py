# core/text_utils.py — Text preprocessing, chunking, and filtering

import re
import config


# Fragments indicating unimportant Docling elements
_JUNK_FRAGMENTS = frozenset([
    "beschr", "abb.", "fig.", "tab.", "quelle", "source",
    "bildplatzhalter", "[image",
])

_JUNK_STARTS = ("[image", "bildplatzhalter")


# Checks if a text element from Docling is irrelevant (page numbers, placeholders, etc.).
def is_text_junk(text: str, label) -> bool:
    """
    Checks if a text element from Docling is irrelevant (page numbers, placeholders, etc.).
    Returns True if the element should be skipped.
    """
    text = text.strip()
    if not text:
        return True

    label_str = str(label).lower()

    # Always keep titles/headers
    if "title" in label_str or "header" in label_str:
        return False

    cleaned = text.lower().replace(":", "").replace(".", "").strip()

    if cleaned in _JUNK_FRAGMENTS:
        return True
    if any(text.lower().startswith(x) for x in _JUNK_STARTS):
        return True

    # Pure number < 1000 → page number
    if text.isdigit() and int(text) < 1000:
        return True

    # Fewer than 2 words (except title/header, already handled above)
    if len(text.split()) < 2:
        return True

    return False


# Converts Docling label to Markdown header prefix.
def inject_markdown_headers(text: str, label_str: str) -> str:
    """
    Converts Docling label to Markdown header prefix.
    Order: check section before header, since "section_header" would otherwise match incorrectly.
    """
    if "section" in label_str:
        return f"\n### {text}"
    if "header" in label_str:
        return f"\n## {text}"
    return text


# Splits long text into chunks with soft boundaries.
def smart_split_text(text: str, block_size: int = None) -> list[str]:
    """
    Splits long text into chunks with soft boundaries.
    Respects paragraph boundaries, falls back to sentence boundaries.
    """
    if block_size is None:
        block_size = config.CHUNK_SIZE_CHARS

    blocks = []
    paragraphs = text.split("\n\n")
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > block_size:
            # Finish current chunk
            if current_chunk:
                blocks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) > block_size:
                    if sub_chunk:
                        blocks.append(sub_chunk.strip())
                    sub_chunk = sent
                else:
                    sub_chunk = (sub_chunk + " " + sent).strip() if sub_chunk else sent
            if sub_chunk:
                current_chunk.append(sub_chunk)
                current_len = len(sub_chunk)

        elif current_len + para_len < block_size:
            current_chunk.append(para)
            current_len += para_len + 2

        else:
            if current_chunk:
                blocks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len

    if current_chunk:
        blocks.append("\n\n".join(current_chunk))

    return [b for b in blocks if b.strip()]


# Cleans LLM output: removes code block wrappers, normalizes blank lines.
def clean_llm_markdown_output(text: str) -> str:
    """
    Cleans LLM output: removes code block wrappers, normalizes blank lines.
    """
    text = text.strip()
    # Remove ```markdown ... ``` or ``` ... ``` wrappers
    text = re.sub(r'^```(?:markdown)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    text = text.strip()
    # More than 2 consecutive blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# Extracts last N characters from text.
def extract_last_n_chars(text: str, n: int) -> str:
    return text[-n:] if len(text) > n else text


# Extracts first N characters from text.
def extract_first_n_chars(text: str, n: int) -> str:
    return text[:n] if len(text) > n else text


# Splits text into overlapping chunks for RAG indexing.
def chunk_text_with_overlap(text: str, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """
    Splits text into overlapping chunks for RAG indexing.
    Returns list of {'text': str, 'start': int, 'end': int}.
    """
    if chunk_size is None:
        chunk_size = config.RAG_CHUNK_SIZE
    if overlap is None:
        overlap = config.RAG_CHUNK_OVERLAP

    if not text.strip():
        return []

    # First split by sentences to find natural boundaries
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)

    chunks = []
    current = ""
    current_start = 0
    pos = 0

    for sent in sentences:
        sent_len = len(sent)

        if len(current) + sent_len > chunk_size and current:
            chunks.append({
                "text": current.strip(),
                "start": current_start,
                "end": current_start + len(current)
            })
            # Overlap: use last `overlap` characters as basis for next chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current_start = current_start + len(current) - len(overlap_text)
            current = overlap_text + " " + sent
        else:
            if not current:
                current_start = pos
            current = (current + " " + sent).strip() if current else sent

        pos += sent_len + 1  # +1 for separator

    if current.strip():
        chunks.append({
            "text": current.strip(),
            "start": current_start,
            "end": current_start + len(current)
        })

    return chunks
