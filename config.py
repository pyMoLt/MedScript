# config.py — Centralized configuration constants
# All constants defined centrally. No magic numbers in other modules.
# IMPORTANT: All modules use `import config` then `config.VALUE`.
# NEVER use `from config import VALUE` — this prevents runtime patching
# by settings_manager.py.

from pathlib import Path

# ── LM Studio ─────────────────────────────────────────────────────────────────
LM_STUDIO_HOST        = "127.0.0.1"
LM_STUDIO_PORT        = "1234"
LM_STUDIO_BASE_URL    = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"
LM_STUDIO_API_KEY     = "lm-studio"
LM_STUDIO_TIMEOUT     = 1500.0  # Seconds (25 minutes per call)

# ── Model identifiers ─────────────────────────────────────────────────────────
VISION_MODEL_SEARCH   = "qwen3.6-35b-a3b-ud-mlx"
VISION_MODEL_LOAD     = "qwen3.6-35b-a3b-ud-mlx"

TEXT_MODEL_SEARCH     = "qwen3.6-35b-a3b-ud-mlx"
TEXT_MODEL_LOAD       = "qwen3.6-35b-a3b-ud-mlx"

EMBEDDING_MODEL_ID    = "text-embedding-qwen3-embedding-0.6b"

# ── Processing parameters ─────────────────────────────────────────────────────
CHUNK_SIZE_CHARS      = 3500
CONTEXT_CHARS         = 800
MIN_PIXEL_SIDE        = 250
TARGET_MAX_SIDE       = 2048
JPEG_QUALITY          = 85

# ── Output paths ───────────────────────────────────────────────────────────────
DESKTOP_PATH          = Path.home() / "Desktop"
CENTRAL_OUTPUT_DIR    = DESKTOP_PATH / "Vorlesung_Lernskripte"

# ── RAG configuration ──────────────────────────────────────────────────────────
RAG_BACKEND           = "chroma"        # "chroma" | "qdrant"
RAG_STORES_DIR        = Path(__file__).parent / "data" / "rag_stores"
RAG_TOP_K             = 4
RAG_MIN_SCORE         = 0.55
RAG_IMAGE_MIN_SCORE   = 0.38    # Lower threshold for image searches (cross-domain embedding)
RAG_CHUNK_SIZE        = 600
RAG_CHUNK_OVERLAP     = 80
RAG_IMAGE_DESC_TOKENS = 300
RAG_MAX_IMAGE_INSERTS_PER_SECTION = 3   # Max RAG images per section (3 = 2 good + 1 fallback)

# Qdrant-specific settings
QDRANT_HOST           = "localhost"
QDRANT_PORT           = 6333

# ── Writing parameters ─────────────────────────────────────────────────────────
WRITING_TEMPERATURE        = 0.1    # Temperature for all writing calls
WRITING_MAX_TOKENS         = 32000  # Max tokens including thinking (Qwen3: ~8k-16k thinking + body)
ANALYSIS_MAX_TOKENS        = 10000  # Max tokens for JSON-producing analysis calls
KB_SUBAGENT_MAX_TOKENS     = 8000   # Max tokens for knowledge base compression (thinking overhead!)
WEB_SUBAGENT_MAX_TOKENS    = 12000  # Max tokens for web-search compression subagent (multi-iteration, needs headroom)
FETCH_SUBAGENT_MAX_TOKENS  = 8000   # Max tokens for URL-fetch compression subagent
UTILITY_MAX_TOKENS         = 8000   # Max tokens for short internal calls (title, digest, labels)

# ── Agent-based tool use ───────────────────────────────────────────────────────
AGENT_MAX_TOOL_ITERATIONS  = 8
AGENT_TOOLS_ENABLED        = True
WEB_SUBAGENT_MAX_ITERATIONS   = 5   # Max iterations of web search subagent
FETCH_SUBAGENT_MAX_ITERATIONS = 1   # Max iterations of URL fetch subagent

# ── Deep synthesis image placement ──────────────────────────────────────────────
SYNTHESIS_IMAGE_PLACEMENT_THRESHOLD = 0.6   # Cosine threshold for phase-5 placement
SYNTHESIS_VISUAL_BRIDGE_MAX_TOKENS  = 300   # Max tokens for visual bridge image descriptions
SYNTHESIS_MIN_BLOCK_CHARS           = 300   # Min characters per chapter block (smaller ones are merged)

# ── SearxNG web search ─────────────────────────────────────────────────────────
SEARXNG_BASE_URL           = "http://localhost:8080"
SEARXNG_ENABLED            = True
SEARXNG_MIN_DELAY_SECONDS  = 12.0
SEARXNG_MAX_CALLS_PER_RUN  = 10
SEARXNG_RESULTS_PER_QUERY  = 3
SEARXNG_TIMEOUT_SECONDS    = 10
SEARXNG_JITTER_MIN         = 4.0    # Minimum jitter delay between web calls (seconds)
SEARXNG_JITTER_MAX         = 8.0    # Maximum jitter delay between web calls (seconds)

# ── URL fetch tool ────────────────────────────────────────────────────────────
URL_FETCH_ENABLED               = True
URL_FETCH_MAX_CHARS             = 12000
URL_FETCH_MAX_CALLS_PER_SECTION = 2
URL_FETCH_TIMEOUT_SECONDS       = 15
URL_FETCH_MIN_DELAY_SECONDS     = 3.0

# ── GUI settings ───────────────────────────────────────────────────────────────
GUI_WINDOW_TITLE      = "MedSkript — Medizinisches Lernskript Generator"
GUI_WINDOW_SIZE       = "1200x800"
GUI_LOG_MAX_LINES     = 500
GUI_PREVIEW_REFRESH   = 2000

# ── Persistent user settings ───────────────────────────────────────────────────
USER_SETTINGS_FILE    = Path.home() / ".medskript" / "settings.json"

# Keys that can be overridden via UserSettings (whitelist)
PERSISTABLE_KEYS: set = {
    "LM_STUDIO_HOST", "LM_STUDIO_PORT", "LM_STUDIO_TIMEOUT",
    "VISION_MODEL_SEARCH", "VISION_MODEL_LOAD",
    "TEXT_MODEL_SEARCH",   "TEXT_MODEL_LOAD",
    "EMBEDDING_MODEL_ID",
    # Websuche
    "SEARXNG_BASE_URL", "SEARXNG_ENABLED",
    "SEARXNG_MIN_DELAY_SECONDS", "SEARXNG_MAX_CALLS_PER_RUN",
    "SEARXNG_RESULTS_PER_QUERY",
    "SEARXNG_JITTER_MIN", "SEARXNG_JITTER_MAX",
    "URL_FETCH_ENABLED", "URL_FETCH_MAX_CHARS",
    "URL_FETCH_MAX_CALLS_PER_SECTION",
    "WEB_SUBAGENT_MAX_ITERATIONS", "FETCH_SUBAGENT_MAX_ITERATIONS",
    # RAG
    "RAG_BACKEND", "QDRANT_HOST", "QDRANT_PORT",
    "RAG_TOP_K", "RAG_MIN_SCORE", "RAG_IMAGE_MIN_SCORE",
    "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP", "RAG_IMAGE_DESC_TOKENS",
    "RAG_MAX_IMAGE_INSERTS_PER_SECTION",
    # Schreiben
    "WRITING_TEMPERATURE", "WRITING_MAX_TOKENS",
    "ANALYSIS_MAX_TOKENS", "KB_SUBAGENT_MAX_TOKENS",
    "WEB_SUBAGENT_MAX_TOKENS", "FETCH_SUBAGENT_MAX_TOKENS", "UTILITY_MAX_TOKENS",
    "AGENT_MAX_TOOL_ITERATIONS", "AGENT_TOOLS_ENABLED",
    # Output
    "CENTRAL_OUTPUT_DIR",
    # Deep Synthesis
    "SYNTHESIS_IMAGE_PLACEMENT_THRESHOLD", "SYNTHESIS_VISUAL_BRIDGE_MAX_TOKENS",
    "SYNTHESIS_MIN_BLOCK_CHARS",
    # OCR-Modus
    "OCR_MODE_DEFAULT",
    "OCR_ANALYSIS_DPI", "OCR_WRITING_DPI",
    "OCR_MAX_PAGES_PER_WRITING_CALL", "OCR_MAX_PAGES_PER_ANALYSIS_BATCH",
    "OCR_ANALYSIS_CACHE_ENABLED", "OCR_ANALYSIS_CACHE_DIR", "OCR_WRITING_CACHE_DIR",
    "OCR_ROLLING_CONTEXT_CHARS", "OCR_MAX_FIGURES_PER_SECTION",
    "OCR_FIGURE_JPEG_QUALITY", "OCR_FALLBACK_GROUP_SIZE",
    "OCR_DIGEST_INTERVAL",
}


# ── OCR mode (image-based processing mode) ─────────────────────────────────────
OCR_MODE_DEFAULT                 = False   # Default: text-chunk mode
OCR_ANALYSIS_DPI                 = 96      # DPI for analysis-pass thumbnails
OCR_WRITING_DPI                  = 130     # DPI for writing-pass page images
OCR_MAX_PAGES_PER_WRITING_CALL   = 12      # Max pages per writing call
OCR_MAX_PAGES_PER_ANALYSIS_BATCH = 22      # Max thumbnails per analysis call
OCR_ANALYSIS_CACHE_ENABLED       = True    # Cache analysis-pass JSON
OCR_ANALYSIS_CACHE_DIR           = Path.home() / ".medskript" / "cache" / "ocr_analysis"
OCR_WRITING_CACHE_DIR            = Path.home() / ".medskript" / "cache" / "ocr_writing"
OCR_ROLLING_CONTEXT_CHARS        = 400     # Characters of last section as context
OCR_MAX_FIGURES_PER_SECTION      = 8       # Max Docling figures offered per section
OCR_FIGURE_JPEG_QUALITY          = 88      # JPEG quality of saved figures
OCR_FALLBACK_GROUP_SIZE          = 10      # Fallback pages/group when analysis-pass fails
OCR_DIGEST_INTERVAL              = 5       # Generate global context digest every N sections
OCR_SKIP_TOPICS: set             = {       # Topic names to skip during writing
    "titelseite", "leer", "inhalt", "agenda", "gliederung", "inhaltsverzeichnis"
}

# ── Deadlock prevention (set in main.py as os.environ) ──────────────────────────
# OMP_NUM_THREADS=1, MKL_NUM_THREADS=1
# OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# TOKENIZERS_PARALLELISM=false
