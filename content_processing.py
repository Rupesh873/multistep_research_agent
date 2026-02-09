from typing import Dict, Any, List
import re


BOILERPLATE_PATTERNS = [
    r"skip to main content",
    r"we gratefully acknowledge",
    r"\bdonate\b",
    r"advanced search",
    r"\bcomments?\b",
    r"copyright",
    r"terms of use",
    r"privacy policy",
    r"subscribe",
    r"newsletter",
]


def remove_boilerplate(text: str) -> str:
    # DO NOT lowercase everything; preserve casing
    out = text or ""
    for pattern in BOILERPLATE_PATTERNS:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    return out


def normalize_text(text: str) -> str:
    text = remove_boilerplate(text)
    text = re.sub(r"\[[0-9]+\]", "", text)  # citation markers
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 380) -> List[str]:
    words = (text or "").split()
    chunks: List[str] = []
    current: List[str] = []

    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def quality_filter(text: str) -> bool:
    if not text or len(text) < 450:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.60:
        return False

    return True


def content_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    processed_chunks: List[Dict[str, Any]] = []

    for item in state.get("scraped_content", []):
        source = item.get("source")
        url = item.get("url")
        content_type = item.get("type", "html")

        raw_text = item.get("content", "")
        if not raw_text or len(raw_text) < 350:
            continue

        clean_text = normalize_text(raw_text)
        chunks = chunk_text(clean_text)

        for idx, chunk in enumerate(chunks):
            if not quality_filter(chunk):
                continue

            processed_chunks.append({
                "source": source,
                "url": url,
                "type": content_type,
                "chunk_id": idx,
                "word_count": len(chunk.split()),
                "text": chunk
            })

    return {"processed_chunks": processed_chunks}

