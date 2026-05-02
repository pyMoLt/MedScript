# rag/url_fetcher.py — Fetches URL and extracts main text content

# Fetches URL, extracts main text (no HTML junk), returns token-limited text.

import re
import config

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Tags komplett entfernen (kein Text daraus)
STRIP_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "advertisement", "iframe", "noscript", "form", "button",
]


def is_url_fetchable(url: str) -> bool:
    """
    Validiert ob eine URL fetchbar ist.
    Blockiert lokale Netz-Adressen (SSRF-Schutz).
    """
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower().strip()
    # Nur HTTP/HTTPS
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        return False
    # Lokale Adressen blockieren
    blocked_patterns = [
        "localhost", "127.", "192.168.", "10.", "172.16.", "172.17.", "172.18.",
        "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
        "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.", "::1",
        "0.0.0.0",
    ]
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        for blocked in blocked_patterns:
            if host == blocked or host.startswith(blocked):
                return False
    except Exception:
        return False
    return True


def fetch_url_content(url: str, max_chars: int = None) -> dict:
    """
    Ruft URL ab und extrahiert Plaintext.
    Gibt immer ein dict zurück (wirft nie).
    """
    if max_chars is None:
        max_chars = config.URL_FETCH_MAX_CHARS

    if not REQUESTS_AVAILABLE:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": "requests nicht installiert"}
    if not BS4_AVAILABLE:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": "beautifulsoup4 nicht installiert"}

    if not is_url_fetchable(url):
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": "URL nicht erlaubt (lokales Netz oder ungültiges Schema)"}

    try:
        resp = requests.get(
            url,
            timeout=config.URL_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "MedSkript/1.0 (Educational)"},
        )
    except requests.exceptions.Timeout:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": str(e)}

    if resp.status_code != 200:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": f"HTTP {resp.status_code}"}

    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and "text/plain" not in content_type:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": f"Unbekannter Content-Type: {content_type}"}

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Unerwünschte Tags entfernen
        for tag in STRIP_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Titel extrahieren
        title_el = soup.find("title")
        title = title_el.get_text(strip=True) if title_el else ""

        # Hauptinhalt finden (Priorität-Liste)
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find(attrs={"role": "main"}) or
            soup.find("div", class_="content") or
            soup.find("div", id="content") or
            soup.find("body")
        )
        if main is None:
            main = soup

        raw_text = main.get_text(separator="\n", strip=True)
    except Exception as e:
        return {"success": False, "url": url, "title": "", "text": "",
                "truncated": False, "error": f"Parse-Fehler: {e}"}

    # Bereinigung
    lines = raw_text.split("\n")
    lines = [l for l in lines if len(l.strip()) >= 3]  # UI-Artefakte entfernen
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    truncated = len(text) > max_chars
    if truncated:
        # Bei Satzgrenze kürzen wenn möglich
        cut = text[:max_chars]
        last_dot = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if last_dot > max_chars * 0.8:
            cut = cut[:last_dot + 1]
        text = cut

    return {
        "success": True,
        "url": url,
        "title": title,
        "text": text,
        "truncated": truncated,
        "error": "",
    }


def format_fetched_content_for_llm(fetch_result: dict) -> str:
    """Formatiert fetch_url_content() Ergebnis als LLM-lesbaren String."""
    if not fetch_result.get("success"):
        return f"[FETCH FEHLGESCHLAGEN: {fetch_result['url']} — {fetch_result['error']}]"
    trunc_note = "...(gekürzt)" if fetch_result.get("truncated") else "(vollständig)"
    return (
        f"[WEBSEITE: {fetch_result['title']} | {fetch_result['url']}]\n"
        f"{fetch_result['text']}\n"
        f"[{trunc_note}]"
    )
