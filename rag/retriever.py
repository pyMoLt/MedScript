# rag/retriever.py — Type-safe query interface for both modes

# Type-safe query interface for both modes.
# Primarily called by rag/tools.py (ToolExecutor).

import config
from core.llm_client import get_embedding, create_openai_client
from rag.store import create_store


def _get_client():
    return create_openai_client()


# Performs semantic search in RAG store.
def retrieve(
    query_text: str,
    store_name: str,
    top_k: int = None,
    min_score: float = None,
    include_images: bool = True,
) -> list[dict]:
    """
    Performs semantic search in RAG store.
    Returns list of result dicts. Returns [] on error (never raises).
    """
    if top_k is None:
        top_k = config.RAG_TOP_K
    if min_score is None:
        min_score = config.RAG_MIN_SCORE
    try:
        client = _get_client()
        embedding = get_embedding(client, query_text)
        if embedding is None:
            return []
        store = create_store(store_name)
        results = store.query(embedding, top_k, min_score)
        if not include_images:
            results = [r for r in results if r.get("type") != "image_description"]
        return results
    except Exception as e:
        print(f"❌ retrieve Fehler: {e}")
        return []


# Optimized query for a script section.
def retrieve_for_section(
    section_title: str,
    section_text: str,
    store_name: str,
    top_k: int = None,
) -> list[dict]:
    """
    Optimized query for a script section.
    Filters out results already contained in section_text.
    """
    query = f"{section_title}. {section_text[:200]}"
    results = retrieve(query, store_name, top_k=top_k)
    # Duplikat-Filter: Text der schon im Abschnitt vorkommt weglassen
    filtered = []
    for r in results:
        snippet = r["text"][:80].lower()
        if snippet not in section_text.lower():
            filtered.append(r)
    return filtered


# Batch query for multiple queries at once.
def batch_retrieve(
    queries: list[str],
    store_name: str,
    top_k: int = None,
) -> list[list[dict]]:
    """
    Batch query for multiple queries at once.
    Uses a single client object for all queries.
    """
    if top_k is None:
        top_k = config.RAG_TOP_K
    min_score = config.RAG_MIN_SCORE
    try:
        client = _get_client()
        store = create_store(store_name)
        all_results = []
        for q in queries:
            emb = get_embedding(client, q)
            if emb is None:
                all_results.append([])
                continue
            results = store.query(emb, top_k, min_score)
            all_results.append(results)
        return all_results
    except Exception as e:
        print(f"❌ batch_retrieve Fehler: {e}")
        return [[] for _ in queries]
