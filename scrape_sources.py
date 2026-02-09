import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse


# -----------------------
# Bigger caps (more evidence)
# -----------------------
MAX_WEB_URLS = 30
MAX_NEWS_URLS = 30
MAX_COMMUNITY_URLS = 15
MAX_ARXIV_ITEMS = 15
MAX_CONTENT_READER = 25

MIN_TEXT_CHARS = 800
MAX_STORE_CHARS = 20000

# Block noisy domains that destroy coherence (generic choice: forums/social)
BLOCKED_DOMAINS = {
    "defence.in",
    "reddit.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _looks_like_research_query(user_query: str) -> bool:
    q = (user_query or "").lower()
    triggers = ["paper", "arxiv", "research", "study", "journal", "doi", "preprint", "literature review"]
    return any(t in q for t in triggers)


def _clean_soup(soup: BeautifulSoup) -> None:
    # hard remove non-text
    for s in soup(["script", "style", "noscript", "svg"]):
        s.decompose()

    # remove layout containers
    for s in soup.find_all(["nav", "header", "footer", "aside", "form"]):
        s.decompose()

    # remove common junk by id/class keywords
    junk_keywords = [
        "cookie", "consent", "subscribe", "paywall", "banner",
        "advert", "ads", "promo", "newsletter",
        "share", "social", "follow",
        "comment", "comments", "disqus",
        "related", "recommended", "trending",
        "sidebar", "nav", "menu", "footer", "header",
        "breadcrumb", "modal", "popup",
    ]

    def is_junk(tag) -> bool:
        attrs = " ".join([
            " ".join(tag.get("class", [])) if tag.get("class") else "",
            tag.get("id") or ""
        ]).lower()
        return any(k in attrs for k in junk_keywords)

    for tag in soup.find_all(is_junk):
        tag.decompose()


def _extract_main_text(soup: BeautifulSoup) -> str:
    """
    Prefer <article> or <main>. Fallback to body.
    """
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    # normalize whitespace
    return " ".join(text.split())


def _scrape_html(url: str, source: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=18)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        _clean_soup(soup)
        text = _extract_main_text(soup)

        if not text or len(text) < MIN_TEXT_CHARS:
            return None

        return {
            "source": source,
            "url": url,
            "type": "html",
            "content": text[:MAX_STORE_CHARS],
        }
    except Exception:
        return None


def scrape_sources_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scraped: List[Dict[str, Any]] = []
    user_query = (state.get("user_query") or "").strip()

    # 1) Web search URLs
    for url in (state.get("web_search") or [])[:MAX_WEB_URLS]:
        if not isinstance(url, str):
            continue
        d = _domain(url)
        if d in BLOCKED_DOMAINS:
            continue
        item = _scrape_html(url, "web_search")
        if item:
            scraped.append(item)

    # 2) News URLs
    for item in (state.get("news_trend") or [])[:MAX_NEWS_URLS]:
        if not isinstance(item, str):
            continue
        if ":" in item:
            _, url = item.split(":", 1)
            url = url.strip()
        else:
            url = item.strip()

        d = _domain(url)
        if d in BLOCKED_DOMAINS:
            continue

        scraped_item = _scrape_html(url, "news_trend")
        if scraped_item:
            scraped.append(scraped_item)

    # 3) Community URLs
    for c in (state.get("community_insight") or [])[:MAX_COMMUNITY_URLS]:
        if not isinstance(c, dict):
            continue
        url = (c.get("url") or "").strip()
        if not url:
            continue
        d = _domain(url)
        if d in BLOCKED_DOMAINS:
            continue
        item = _scrape_html(url, "community_insight")
        if item:
            scraped.append(item)

    # 4) Authority source (arXiv metadata) — only if query looks research-like
    if _looks_like_research_query(user_query):
        for p in (state.get("authority_source") or [])[:MAX_ARXIV_ITEMS]:
            if not isinstance(p, dict):
                continue
            abstract = (p.get("abstract") or "").strip()
            if len(abstract) < 200:
                continue
            scraped.append({
                "source": "authority_source",
                "url": p.get("url"),
                "title": p.get("title"),
                "type": "arxiv_metadata",
                "content": abstract[:MAX_STORE_CHARS],
            })

    # 5) Content reader outputs
    content_reader = state.get("content_reader", {})
    if isinstance(content_reader, dict):
        for url, text in list(content_reader.items())[:MAX_CONTENT_READER]:
            url = (url or "").strip()
            text = (text or "").strip()
            if not url or not text or text == "Failed":
                continue
            if len(text) < MIN_TEXT_CHARS:
                continue
            d = _domain(url)
            if d in BLOCKED_DOMAINS:
                continue
            scraped.append({
                "source": "content_reader",
                "url": url,
                "type": "html_extracted",
                "content": text[:MAX_STORE_CHARS],
            })

    return {"scraped_content": scraped}
