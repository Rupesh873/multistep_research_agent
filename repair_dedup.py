# repair_dedup.py
from typing import Dict, Any, List
import hashlib


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def repair_and_dedup_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 6: Repair, Deduplicate, Normalize Evidence

    - NO LLM
    - NO generation
    - LangGraph-safe
    """

    evidence: List[Dict[str, Any]] = state.get("evidence", [])

    seen_hashes = set()
    cleaned_evidence: List[Dict[str, Any]] = []
    dropped = 0

    for item in evidence:
        source = item.get("source")
        data = item.get("data")

        if not source or not data:
            dropped += 1
            continue

        text = str(data).strip()
        if not text:
            dropped += 1
            continue

        h = _hash_text(text)
        if h in seen_hashes:
            dropped += 1
            continue

        seen_hashes.add(h)
        cleaned_evidence.append({
            "source": source,
            "content": text
        })

    return {
        "clean_evidence": cleaned_evidence,
        "repair_stats": {
            "input_items": len(evidence),
            "kept_items": len(cleaned_evidence),
            "dropped_items": dropped
        }
    }


