# rag/web_search.py — SearxNG client with robust error handling

# SearxNG client with robust error handling.
# No rate-limiting here — it's in ToolExecutor.

import config

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def is_searxng_available() -> bool:
    """Schneller Ping-Check ob SearxNG-Server erreichbar ist."""
    if not REQUESTS_AVAILABLE:
        return False
    try:
        url = f"{config.SEARXNG_BASE_URL}/search"
        resp = requests.get(url, params={"q": "test", "format": "json"}, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def searxng_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Führt Websuche über SearxNG durch.
    Gibt Liste von {'title', 'url', 'snippet'} zurück.
    Stateless — kein Rate-Limiting.
    """
    if not REQUESTS_AVAILABLE:
        return []
    try:
        resp = requests.get(
            f"{config.SEARXNG_BASE_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general,science",
                "language": "de-DE",
                "safesearch": "1",
            },
            timeout=config.SEARXNG_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print(f"⚠️ SearxNG nicht erreichbar.")
        return []
    except requests.exceptions.Timeout:
        print(f"⚠️ SearxNG Timeout.")
        return []
    except Exception as e:
        print(f"⚠️ SearxNG Fehler: {e}")
        return []

    results = []
    for item in data.get("results", [])[:num_results]:
        snippet = item.get("content", item.get("snippet", ""))
        if not snippet:
            continue
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": snippet[:500],  # Max 500 Zeichen
        })
    return results


def format_search_results_for_llm(results: list[dict]) -> str:
    """Formatiert SearxNG-Ergebnisse als lesbaren String für LLM."""
    if not results:
        return "Keine Websuche-Ergebnisse."
    parts = []
    for r in results:
        parts.append(f"[{r['title']} | {r['url']}]\n{r['snippet']}")
    return "\n---\n".join(parts)
