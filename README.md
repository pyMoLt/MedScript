# MedScript

A fully local AI pipeline that converts medical lecture slides (PDF) into structured,
readable learning scripts — no cloud API, no subscriptions, everything runs on your machine.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running MedSkript](#running-medskript)
4. [How It Works](#how-it-works)
5. [Four Processing Modes](#four-processing-modes)
6. [Agentic Writing System](#agentic-writing-system)
7. [RAG Knowledge Base](#rag-knowledge-base)
8. [Configuration](#configuration)
9. [Architecture Overview](#architecture-overview)

---

## Prerequisites

| Component | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | `match` statements require 3.10+; 3.11 recommended |
| [LM Studio](https://lmstudio.ai/) | latest | Must be running before MedSkript starts |
| LLM model | — | Qwen3 recommended (supports tool calls + extended thinking) |
| Embedding model | — | `text-embedding-qwen3-embedding-0.6b` or compatible |
| [SearxNG](https://searxng.github.io/searxng/) | optional | Only needed for the web search feature |

**Tested model:** `qwen3.6-35b-a3b-ud-mlx` (Apple Silicon, MLX-quantized). Any model
with OpenAI-compatible tool calls and a reasonable context window will work.

**macOS — WeasyPrint requires Pango:**
```bash
brew install pango
```

**SearxNG (optional, via Docker):**
```bash
docker run -d -p 8080:8080 searxng/searxng
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/pyMoLt/MedScript.git
cd MedScript

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start LM Studio, load your model, and enable the local server
#    (LM Studio → Settings → Local Server → Start)
```

> On first launch MedSkript creates `~/.medskript/settings.json` with defaults that
> you can change from the GUI Settings tab.

---

## Running MedSkript

### GUI (recommended)

```bash
python main.py
```

On macOS you can also double-click `MedSkript.command` — it activates the virtual
environment and opens the GUI automatically.

### CLI — Single file

```bash
# Text mode (default — good for clean, machine-readable PDFs)
python main.py lecture.pdf

# OCR mode (better for scanned or image-heavy slides)
python main.py --ocr lecture.pdf
```

### CLI — Multiple files (synthesis)

```bash
# Combine several lectures into one structured document
python main.py --mode synthesis lecture1.pdf lecture2.pdf lecture3.pdf

# OCR synthesis
python main.py --ocr --mode synthesis lecture1.pdf lecture2.pdf
```

### CLI — Build a RAG knowledge base

```bash
python main.py --index \
    --input /path/to/textbooks/ \
    --store-name Cardiology \
    [--rebuild]      # overwrite existing store
    [--no-images]    # skip image indexing (much faster)
```

---

## How It Works

MedSkript runs in three stages:

1. **Parse** — reads your PDF using [Docling](https://github.com/DS4SD/docling)
   (text-based modes) or renders pages to images via PyMuPDF (OCR modes).
2. **Plan** — an LLM analyzes the content and produces a structured topic plan with
   main topics, sub-topics, section order, and (in OCR modes) exact page boundaries.
3. **Write** — a second LLM pass writes each section of the script, and can call tools
   mid-generation to pull in textbook knowledge, generate comparison tables, or search the web.

The finished output is a Markdown document converted to PDF or HTML via WeasyPrint.

---

## Four Processing Modes

### `summary` — Single-file, text-based

Best for well-structured, machine-readable PDFs where Docling can extract clean text.

Docling parses the PDF into text and images → chunks are processed sequentially with
rolling context → images are placed inline using cosine-similarity → exported to PDF.

### `synthesis` — Multi-file, text-based

Best for combining multiple lecture PDFs into one cohesive textbook chapter.

A **Master LLM** reads all content across files and produces a unified chapter plan →
a **Sub-Architect LLM** refines each chapter into semantic sub-sections → the
**Writer LLM** writes each sub-section with rolling context and transition hints
from the previous section.

### `ocr_summary` — Single-file, OCR-based

Best for scanned PDFs, image-heavy slides, or PDFs where text extraction is unreliable.

Pages are rendered to low-res thumbnails → an **Analysis Pass LLM** assigns topic labels
to each page group (cached to disk for re-runs) → a **Writing Pass LLM** receives
full-resolution page images and writes each section → a **Rolling Digest** keeps the LLM
oriented across long documents.

### `ocr_synthesis` — Multi-file, OCR-based

Extends `ocr_summary` with the same two-level planning as `synthesis`:
Analysis pass per file → **Master LLM** unifies topics across files →
**Sub-Architect** assigns fractional page boundaries per sub-section →
Writing loop uses pre-planned boundaries when available, falls back to runtime
planning otherwise.

---

## Agentic Writing System

Every section is written inside an **agent loop** that can call tools mid-generation:

```
LLM generates text
  ↓ tool call detected?
Execute tool → inject result → LLM continues
  ↓ no more tool calls (or iteration limit reached)
Return final text
```

Tool call detection handles both OpenAI native `tool_calls` JSON and XML
`<tool_call>...</tool_call>` fallback format (used by Qwen3). Extended thinking blocks
(`<think>...</think>`) are stripped from all LLM outputs before further processing.

### Available Tools

| Tool | What it does |
|---|---|
| `search_knowledge_base(query)` | Semantic search in the indexed textbook RAG store |
| `insert_rag_image(image_id, caption)` | Inserts a textbook figure referenced by a RAG result |
| `get_structured_comparison(topic)` | Generates a formatted comparison table or classification |
| `search_web(query)` | SearxNG web search (optional, rate-limited) |
| `fetch_url(url)` | Fetches and extracts plain text from a URL |

When `search_web` or `fetch_url` is called, a lightweight sub-agent compresses the raw
results into a concise summary before they reach the writing LLM — keeping the main
context clean and focused.

---

## RAG Knowledge Base

MedSkript can index your medical textbooks as a local vector database and use them
as a knowledge source during writing.

**Indexing** extracts and chunks all text, then analyzes images with the vision model
and stores their descriptions. Everything is embedded and stored in
`data/rag_stores/<StoreName>/` using ChromaDB (default) or Qdrant.

**At write time**, the LLM calls `search_knowledge_base`, receives the most relevant
chunks, and weaves the information naturally into the text. If a matching figure is found,
a hint prompts the LLM to call `insert_rag_image` and embed it inline.

---

## Configuration

Edit `~/.medskript/settings.json` or use the **Settings** tab in the GUI.

> All code accesses config values as `config.VALUE` — never `from config import VALUE`.
> This is intentional: `settings_manager.py` patches the config module at runtime via
> `setattr()`, and a local import would miss those updates.

### Key Settings

| Key | Default | Description |
|---|---|---|
| `LM_STUDIO_HOST` | `127.0.0.1` | LM Studio host |
| `LM_STUDIO_PORT` | `1234` | LM Studio port |
| `LM_STUDIO_TIMEOUT` | `1500.0` | Request timeout in seconds |
| `VISION_MODEL_SEARCH` | `qwen3.6-35b-a3b-ud-mlx` | Model ID (text + vision) |
| `EMBEDDING_MODEL_ID` | `text-embedding-qwen3-embedding-0.6b` | Embedding model ID |
| `CENTRAL_OUTPUT_DIR` | `~/Desktop/Vorlesung_Lernskripte` | Output folder |
| `SEARXNG_BASE_URL` | `http://localhost:8080` | SearxNG URL |
| `SEARXNG_ENABLED` | `True` | Enable web search |
| `RAG_BACKEND` | `chroma` | `"chroma"` or `"qdrant"` |
| `OCR_MODE_DEFAULT` | `False` | Use OCR mode by default |
| `OCR_ANALYSIS_DPI` | `96` | DPI for analysis thumbnails |
| `OCR_WRITING_DPI` | `130` | DPI for writing page images |
| `WRITING_TEMPERATURE` | `0.1` | LLM temperature (low = deterministic) |
| `AGENT_MAX_TOOL_ITERATIONS` | `8` | Max tool-call rounds per section |

The complete list of configurable keys is in `config.py` under `PERSISTABLE_KEYS`.

---

## Architecture Overview

```
medskript/
│
├── main.py                  # Entry point — routes to GUI, CLI, or indexing
├── config.py                # All constants (always import config, never from config import)
├── requirements.txt
│
├── core/                    # Shared foundation modules
│   ├── llm_client.py        # LM Studio API, agentic tool-call loop, model switching
│   ├── pdf_parser.py        # Docling PDF parser — text, tables, images
│   ├── page_renderer.py     # PyMuPDF: PDF pages → PIL images (OCR modes)
│   ├── text_utils.py        # Chunking, junk filtering, Markdown cleanup
│   ├── image_utils.py       # Resize, Base64 encoding, perceptual deduplication
│   ├── output.py            # Markdown → PDF/HTML via WeasyPrint, TOC generation
│   ├── cache.py             # Persistent JSON disk cache (OCR analysis results)
│   └── settings_manager.py  # Loads settings.json and patches config at runtime
│
├── modes/                   # The four processing pipelines
│   ├── summary.py           # Single-file text mode
│   ├── synthesis.py         # Multi-file text mode
│   ├── ocr_summary.py       # Single-file OCR mode
│   └── ocr_synthesis.py     # Multi-file OCR mode
│
├── rag/                     # Retrieval-Augmented Generation
│   ├── store.py             # ChromaDB / Qdrant backend abstraction
│   ├── indexer.py           # Textbook indexing pipeline
│   ├── retriever.py         # Semantic search interface
│   ├── tools.py             # ToolExecutor for text modes
│   ├── ocr_tools.py         # ToolExecutor for OCR modes
│   ├── augmenter.py         # Builds tool-instruction blocks for LLM system prompts
│   ├── web_search.py        # SearxNG client
│   └── url_fetcher.py       # URL fetcher with SSRF protection
│
├── gui/                     # Tkinter desktop GUI
│   ├── app.py               # Main window
│   ├── styles.py            # Colors, fonts, layout constants
│   ├── panels/              # File selection, settings, progress log, live preview
│   └── dialogs/             # Advanced settings dialog
│
└── assets/css/              # CSS stylesheets for HTML/PDF output
    ├── default.css
    └── compact.css
```
