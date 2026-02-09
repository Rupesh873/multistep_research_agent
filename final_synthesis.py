# final_synthesis.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import os
import re
import json
import math
from collections import Counter

# Load .env file explicitly (your requirement)
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# LangChain OpenAI
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


# ---------------------------
# ENV / LLM configuration
# ---------------------------
ENV_FILE_NAME = "multistep_research_agent.env"
OPENAI_KEY_ENV = "OPENAI_API_KEY"

# You said: use gpt-4o
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0


# ---------------------------
# Output size knobs (your requirement)
# ---------------------------
TOP_K_CHUNKS = 260
EXCERPT_CHAR_LIMIT = 4200

# REQUIREMENT: 9 to 14 paragraphs (best-effort)
MIN_PARAS = 9
MAX_PARAS = 14

# REQUIREMENT: each paragraph 7 to 13 sentences (best-effort)
MIN_SENTS_PER_PARA = 7
MAX_SENTS_PER_PARA = 13

# Enough total sentences to support 9–14 paras
SENTS_TOTAL_TARGET = 140

# allow more refs
MAX_UNIQUE_REFS = 60

# ---------------------------
# Cleaning knobs
# ---------------------------
# Lowered to allow more sentence candidates (helps you hit 9+ paras)
MIN_SENT_LEN = 28
MAX_SENT_LEN = 480

# Higher threshold => dedupe only near-identical, keeps more content
DUP_SIM_THRESHOLD = 0.92

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")

_BOILERPLATE_RE = re.compile(
    r"(skip to main content|cookie|privacy policy|terms of use|sign in|subscribe|advertisement|"
    r"all rights reserved|click to share|open in new window|related articles|newsletter|"
    r"sponsor message|read more|show more|menu|breadcrumb|login|register)",
    re.IGNORECASE,
)

# common cp1252 -> utf8 artifacts
_MOJIBAKE_MAP = {
    "â€™": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
    "â€˜": "'",
    "â€": '"',
    "Ã©": "é",
    "Ã¨": "è",
    "Ã": "",
}


def _load_env_if_needed() -> None:
    if load_dotenv is None:
        return

    if os.path.exists(ENV_FILE_NAME):
        load_dotenv(ENV_FILE_NAME, override=False)
        return

    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, ENV_FILE_NAME)
    if os.path.exists(candidate):
        load_dotenv(candidate, override=False)


def _fix_mojibake(s: str) -> str:
    if not s:
        return ""
    for k, v in _MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    return s


def _normalize_space(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _score_text_overlap(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    tq = Counter(q)
    tt = Counter(t)
    score = 0.0
    for w, c in tq.items():
        if w in tt:
            score += (1.0 + math.log(1 + tt[w])) * (1.0 + math.log(1 + c))
    return score


def _clean_sentence(s: str) -> str:
    s = _fix_mojibake(_normalize_space(s))
    s = re.sub(r"^[\-\–\—•\*\|\>\:]+\s*", "", s).strip()
    s = _normalize_space(s)
    s = s.replace(" | ", " ")
    s = _normalize_space(s)

    s = re.sub(r"[.]{2,}", ".", s)
    s = re.sub(r"[\!]{2,}", "!", s)
    s = re.sub(r"[\?]{2,}", "?", s)

    if s and s[-1] not in ".!?":
        s += "."

    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]

    return s


def _looks_like_junk(s: str) -> bool:
    if not s:
        return True
    if _BOILERPLATE_RE.search(s):
        return True
    if s.count("|") >= 3 or s.count("›") >= 2 or s.count("»") >= 2:
        return True
    low = s.lower()
    if "reactions" in low and "messages" in low:
        return True
    if "member" in low and "messages" in low:
        return True
    return False


def _looks_like_fragment(s: str) -> bool:
    if not s:
        return True
    if len(s) < MIN_SENT_LEN or len(s) > MAX_SENT_LEN:
        return True
    if _looks_like_junk(s):
        return True

    toks = re.findall(r"[A-Za-z]+", s)
    if len(toks) < 5:
        return True

    if s[-1] not in ".!?":
        return True

    return False


def _split_into_sentences(text: str) -> List[str]:
    text = _fix_mojibake(_normalize_space(text))
    if not text:
        return []
    raw = _SENT_SPLIT.split(text)
    out: List[str] = []
    for s in raw:
        s2 = _clean_sentence(s)
        if _looks_like_fragment(s2):
            continue
        out.append(s2)
    return out


def _dedupe_sentences(items: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_tokens: List[List[str]] = []
    for it in items:
        t = _tokenize(it["sentence"])
        dup = False
        for kt in kept_tokens:
            if _jaccard(t, kt) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(it)
            kept_tokens.append(t)
    return kept


def _select_top_chunks(user_query: str, vector_index: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ch in vector_index:
        txt = (ch.get("text") or ch.get("content") or "")
        if not txt:
            continue
        txt2 = txt[: min(len(txt), 4000)]
        scored.append((_score_text_overlap(user_query, txt2), ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [c for _, c in scored[:k]]
    return vector_index[:k] if vector_index else []


def _mine_candidate_sentences(top_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for ch in top_chunks:
        url = (ch.get("url") or "").strip()
        source = (ch.get("source") or "unknown").strip()
        txt = (ch.get("text") or ch.get("content") or "").strip()
        if not txt:
            continue
        if len(txt) > EXCERPT_CHAR_LIMIT:
            txt = txt[:EXCERPT_CHAR_LIMIT]
        sents = _split_into_sentences(txt)
        for s in sents:
            candidates.append({"sentence": s, "url": url, "source": source})
    return candidates


def _rank_sentences(user_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in candidates:
        scored.append((_score_text_overlap(user_query, it["sentence"]), it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored]


def _assign_ref_ids(items: List[Dict[str, Any]], max_unique_refs: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    url_to_id: Dict[str, int] = {}
    refs: List[str] = []
    filtered: List[Dict[str, Any]] = []

    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            filtered.append(it)
            continue

        if url not in url_to_id:
            if len(refs) >= max_unique_refs:
                # Drop items with NEW urls after cap
                continue
            url_to_id[url] = len(refs) + 1
            refs.append(url)

        it["ref_id"] = url_to_id[url]
        filtered.append(it)

    return filtered, refs


def _get_llm() -> Optional[Any]:
    _load_env_if_needed()
    if ChatOpenAI is None:
        return None
    if not os.getenv(OPENAI_KEY_ENV):
        return None
    try:
        return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    except Exception:
        return None


def _safe_json_extract(txt: str) -> Optional[dict]:
    txt = (txt or "").strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _llm_pick_ids(
    user_query: str,
    rows: List[Dict[str, Any]],
) -> Optional[List[List[int]]]:
    """
    LLM creates an outline (paragraphs of sentence IDs).
    We then validate / repair it to satisfy 9–14 paras, 7–13 sents each.
    """
    llm = _get_llm()
    if llm is None:
        return None

    # Keep prompt size bounded
    max_send = min(len(rows), max(SENTS_TOTAL_TARGET * 5, 650))
    rows = rows[:max_send]

    listing = []
    for r in rows:
        sid = r["sid"]
        rid = r.get("ref_id")
        listing.append(f"{sid} | ref={rid if isinstance(rid, int) else 0} | {r['sentence']}")

    max_possible_total = min(len(rows), MAX_PARAS * MAX_SENTS_PER_PARA)
    desired_total = min(SENTS_TOTAL_TARGET, max_possible_total)

    sys = (
        "You are a STRICT evidence selector and organizer.\n"
        "You MUST ONLY output JSON.\n"
        "You MUST NOT invent any facts.\n"
        "Do not repeat sentence IDs.\n"
        "Prefer creating many distinct paragraphs.\n"
    )

    user = (
        f"User query:\n{user_query}\n\n"
        f"Constraints:\n"
        f"- Paragraphs: {MIN_PARAS} to {MAX_PARAS}\n"
        f"- Sentences per paragraph: {MIN_SENTS_PER_PARA} to {MAX_SENTS_PER_PARA}\n"
        f"- Try to use about {desired_total} sentences total.\n"
        f"- Do NOT repeat IDs.\n\n"
        "Candidates:\n"
        + "\n".join(listing)
        + "\n\nReturn JSON ONLY:\n"
          '{ "paragraphs": [[1,2,3], [4,5,6]] }\n'
    )

    try:
        resp = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        data = _safe_json_extract(resp.content if hasattr(resp, "content") else str(resp))
        if not data:
            return None
        paras = data.get("paragraphs")
        if not isinstance(paras, list):
            return None

        out: List[List[int]] = []
        seen = set()
        for p in paras:
            if not isinstance(p, list):
                continue
            p2 = []
            for sid in p:
                if isinstance(sid, int) and sid not in seen:
                    seen.add(sid)
                    p2.append(sid)
            if p2:
                out.append(p2)
        return out if out else None
    except Exception:
        return None


def _llm_make_heading(user_query: str, paragraph_text: str) -> Optional[str]:
    llm = _get_llm()
    if llm is None:
        return None

    sys = (
        "You generate short grounded headings.\n"
        "Do not add new facts.\n"
        "Return ONLY the heading text.\n"
    )
    user = (
        f"User query: {user_query}\n\n"
        f"Paragraph:\n{paragraph_text}\n\n"
        "Write a concise heading (4–10 words).\n"
    )

    try:
        resp = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        title = _fix_mojibake(_normalize_space(resp.content if hasattr(resp, "content") else str(resp)))
        title = re.sub(r"[\[\]\(\)]", "", title).strip()
        if not title:
            return None
        if title[0].isalpha():
            title = title[0].upper() + title[1:]
        return title
    except Exception:
        return None


def _llm_rewrite_paragraph(
    user_query: str,
    paragraph_items: List[Dict[str, Any]],
    prev_heading: Optional[str] = None,
) -> Optional[str]:
    """
    Paraphrase/summarize ONLY from evidence lines.
    CRITICAL: ensure EACH sentence has at least one [n].
    """
    llm = _get_llm()
    if llm is None:
        return None

    evidence_lines = []
    for it in paragraph_items:
        rid = it.get("ref_id", 0)
        evidence_lines.append(f"[{rid}] {it['sentence']}")

    continuity_hint = ""
    if prev_heading:
        continuity_hint = f"Maintain continuity with the previous theme: {prev_heading}\n"

    sys = (
        "You are a grounded research writer.\n"
        "Rules (STRICT):\n"
        "1) Do NOT invent facts, entities, dates, or numbers.\n"
        "2) Use ONLY the meaning in the EVIDENCE lines.\n"
        "3) You MAY paraphrase/summarize to improve flow.\n"
        "4) EVERY sentence MUST include at least one citation marker like [3].\n"
        "5) Output a single paragraph only.\n"
        "6) Aim for 7–13 sentences if possible.\n"
    )

    user = (
        f"User query: {user_query}\n"
        + continuity_hint
        + "\nEVIDENCE (use only these facts):\n"
        + "\n".join(evidence_lines)
        + "\n\nWrite ONE coherent paragraph.\n"
        "Aim for 7–13 sentences if evidence allows.\n"
        "Every sentence must contain at least one citation like [12].\n"
        "Do not add headings.\n"
    )

    try:
        resp = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        txt = resp.content if hasattr(resp, "content") else str(resp)
        txt = _fix_mojibake(txt)
        txt = txt.replace("\r\n", "\n").strip()
        txt = _normalize_space(txt)

        if not txt:
            return None
        if txt[-1] not in ".!?":
            txt += "."
        return txt
    except Exception:
        return None


def _llm_validate_paragraph(paragraph_text: str, paragraph_items: List[Dict[str, Any]]) -> str:
    llm = _get_llm()
    if llm is None:
        return paragraph_text

    evidence_lines = []
    for it in paragraph_items:
        rid = it.get("ref_id", 0)
        evidence_lines.append(f"[{rid}] {it['sentence']}")

    sys = (
        "You are a strict verifier.\n"
        "Remove any sentence not supported by evidence.\n"
        "Do NOT add new information.\n"
        "Keep citations; ensure every remaining sentence has at least one [n].\n"
        "Return a single paragraph only.\n"
    )

    user = (
        "EVIDENCE:\n" + "\n".join(evidence_lines) + "\n\n"
        "DRAFT:\n" + paragraph_text + "\n\n"
        "Return a cleaned paragraph.\n"
    )

    try:
        resp = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        txt = resp.content if hasattr(resp, "content") else str(resp)
        txt = _fix_mojibake(_normalize_space(txt))
        if txt and txt[-1] not in ".!?":
            txt += "."
        return txt if txt else paragraph_text
    except Exception:
        return paragraph_text


def _llm_write_conclusion(user_query: str, headings_and_paras: List[Tuple[str, str]]) -> Optional[str]:
    llm = _get_llm()
    if llm is None:
        return None

    material = []
    for h, p in headings_and_paras:
        material.append(f"HEADING: {h}\nPARAGRAPH: {p}")

    sys = (
        "You write a grounded conclusion.\n"
        "Do NOT add new facts.\n"
        "Summarize only themes already present.\n"
        "Return ONE short concluding paragraph.\n"
        "No heading.\n"
    )

    user = (
        f"User query: {user_query}\n\n"
        "MATERIAL:\n"
        + "\n\n".join(material)
        + "\n\nWrite a brief conclusion tying the themes together.\n"
    )

    try:
        resp = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        txt = resp.content if hasattr(resp, "content") else str(resp)
        txt = _fix_mojibake(_normalize_space(txt))
        if not txt:
            return None
        if txt[-1] not in ".!?":
            txt += "."
        return txt
    except Exception:
        return None


def _heuristic_outline(rows: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Deterministic fallback outline.
    Enforces 9–14 paragraphs and 7–13 sentences each as best-effort.
    """
    sids = [r["sid"] for r in rows]
    max_total = min(len(sids), MAX_PARAS * MAX_SENTS_PER_PARA, SENTS_TOTAL_TARGET)
    sids = sids[:max_total]

    # Aim for at least MIN_PARAS paragraphs if possible
    # Compute para_count based on available sentences
    possible_min_total = MIN_PARAS * MIN_SENTS_PER_PARA
    if len(sids) >= possible_min_total:
        para_count = MIN_PARAS
        # If we have enough, increase para_count up to MAX_PARAS
        while para_count < MAX_PARAS and len(sids) >= (para_count + 1) * MIN_SENTS_PER_PARA:
            para_count += 1
    else:
        # Not enough evidence; do what we can
        para_count = max(1, min(MAX_PARAS, max_total // max(1, MIN_SENTS_PER_PARA)))

    paras: List[List[int]] = []
    idx = 0

    # Try to allocate MIN_SENTS_PER_PARA each first
    for _ in range(para_count):
        if idx >= len(sids):
            break
        take = min(MIN_SENTS_PER_PARA, len(sids) - idx)
        paras.append(sids[idx: idx + take])
        idx += take

    # Distribute remaining sentences round-robin, not exceeding MAX_SENTS_PER_PARA
    p = 0
    while idx < len(sids) and paras:
        if len(paras[p]) < MAX_SENTS_PER_PARA:
            paras[p].append(sids[idx])
            idx += 1
        p = (p + 1) % len(paras)

    # Drop any tiny paras (<2 sentences)
    paras = [p for p in paras if len(p) >= 2]
    return paras if paras else [sids]


def _repair_outline(rows: List[Dict[str, Any]], outline: List[List[int]]) -> List[List[int]]:
    """
    Repair LLM outline to meet constraints:
    - 9–14 paragraphs best-effort
    - 7–13 sentences per paragraph best-effort
    - no empty paragraphs
    """
    # flatten unique while preserving order
    used = []
    seen = set()
    for para in outline or []:
        if not isinstance(para, list):
            continue
        for sid in para:
            if isinstance(sid, int) and sid not in seen and 1 <= sid <= len(rows):
                seen.add(sid)
                used.append(sid)

    if not used:
        return _heuristic_outline(rows)

    # Cap total
    used = used[: min(len(used), MAX_PARAS * MAX_SENTS_PER_PARA, SENTS_TOTAL_TARGET)]

    # Build paras with min requirement
    paras: List[List[int]] = []
    idx = 0

    # First, build MIN_PARAS paragraphs with MIN_SENTS_PER_PARA if possible
    need_min_total = MIN_PARAS * MIN_SENTS_PER_PARA
    if len(used) >= need_min_total:
        para_count = MIN_PARAS
        while para_count < MAX_PARAS and len(used) >= (para_count + 1) * MIN_SENTS_PER_PARA:
            para_count += 1
    else:
        para_count = max(1, min(MAX_PARAS, len(used) // max(1, MIN_SENTS_PER_PARA)))

    for _ in range(para_count):
        if idx >= len(used):
            break
        take = min(MIN_SENTS_PER_PARA, len(used) - idx)
        paras.append(used[idx: idx + take])
        idx += take

    # distribute remaining without exceeding MAX_SENTS_PER_PARA
    p = 0
    while idx < len(used) and paras:
        if len(paras[p]) < MAX_SENTS_PER_PARA:
            paras[p].append(used[idx])
            idx += 1
        p = (p + 1) % len(paras)

    # Final cleanup
    paras = [p for p in paras if len(p) >= 2]
    if not paras:
        return _heuristic_outline(rows)

    return paras


def _render_answer_no_refs_in_text(
    user_query: str,
    sections: List[Tuple[str, str]],
    conclusion: Optional[str],
) -> str:
    """
    IMPORTANT: We do NOT append References here.
    UI already shows references as links from the separate `references` field.
    """
    lines: List[str] = [f"Research Brief: {user_query.strip()}", ""]

    for heading, para in sections:
        if heading:
            lines.append(heading)
        lines.append(para)
        lines.append("")

    if conclusion:
        lines.append("Conclusion")
        lines.append(conclusion)

    return "\n".join(lines).strip()


def final_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = (state.get("user_query") or "").strip()
    vector_index = state.get("vector_index") or []

    if not user_query:
        return {"final_answer": "Query is empty.", "references": [], "mode": "error"}

    if not vector_index:
        return {"final_answer": "No evidence available (vector_index is empty).", "references": [], "mode": "error"}

    # 1) choose chunks
    top_chunks = _select_top_chunks(user_query, vector_index, k=TOP_K_CHUNKS)
    if not top_chunks:
        return {"final_answer": "No relevant evidence found.", "references": [], "mode": "error"}

    # 2) mine sentences
    candidates = _mine_candidate_sentences(top_chunks)
    if not candidates:
        return {"final_answer": "No usable sentences found in scraped content.", "references": [], "mode": "error"}

    # 3) rank + dedupe (lenient)
    ranked = _rank_sentences(user_query, candidates)
    ranked = _dedupe_sentences(ranked, threshold=DUP_SIM_THRESHOLD)

    # 4) assign refs
    ranked, refs_all = _assign_ref_ids(ranked, max_unique_refs=MAX_UNIQUE_REFS)
    if not ranked:
        return {"final_answer": "No usable evidence after filtering.", "references": [], "mode": "error"}

    # 5) pool
    pool_size = min(len(ranked), max(SENTS_TOTAL_TARGET * 6, 850))
    pool = ranked[:pool_size]
    for i, it in enumerate(pool, start=1):
        it["sid"] = i

    items_by_sid = {it["sid"]: it for it in pool}

    # 6) outline (LLM then repair, else heuristic)
    outline = _llm_pick_ids(user_query=user_query, rows=pool)
    if not outline:
        outline = _heuristic_outline(pool)
    outline = _repair_outline(pool, outline)

    # 7) rewrite paragraphs + headings
    sections_raw: List[Tuple[str, str]] = []
    used_urls_in_order: List[str] = []

    def add_used_url(url: str) -> None:
        if url and url not in used_urls_in_order:
            used_urls_in_order.append(url)

    prev_heading: Optional[str] = None
    seen_paragraph_texts = set()

    for para_sids in outline:
        para_items: List[Dict[str, Any]] = []
        for sid in para_sids:
            it = items_by_sid.get(sid)
            if it:
                para_items.append(it)

        # Need a reasonable amount of evidence per paragraph
        if len(para_items) < 3:
            continue

        rewritten = _llm_rewrite_paragraph(user_query, para_items, prev_heading=prev_heading)

        if not rewritten:
            # extractive fallback with citations
            rewritten = " ".join(
                [
                    x["sentence"] + (f"[{x.get('ref_id')}]" if isinstance(x.get("ref_id"), int) else "")
                    for x in para_items
                ]
            )

        rewritten = _llm_validate_paragraph(rewritten, para_items)
        rewritten_norm = rewritten.strip()

        # HARD dedupe paragraphs (prevents "whole answer repeated")
        if rewritten_norm in seen_paragraph_texts:
            continue
        seen_paragraph_texts.add(rewritten_norm)

        heading = _llm_make_heading(user_query, rewritten) or "Key points"

        # Track used urls by citations present in paragraph
        cited_ids = set(int(x) for x in re.findall(r"\[(\d+)\]", rewritten) if x.isdigit())
        for rid in sorted(cited_ids):
            if 1 <= rid <= len(refs_all):
                add_used_url(refs_all[rid - 1])

        sections_raw.append((heading, rewritten))
        prev_heading = heading

        if len(sections_raw) >= MAX_PARAS:
            break

    if not sections_raw:
        return {"final_answer": "Could not assemble a coherent answer from evidence.", "references": [], "mode": "error"}

    # 8) remap citations to compact numbering [1..k] based on USED urls only
    url_to_newid = {url: i + 1 for i, url in enumerate(used_urls_in_order)}

    def remap_citations(text: str) -> str:
        def repl(m):
            old_id = int(m.group(1))
            if 1 <= old_id <= len(refs_all):
                url = refs_all[old_id - 1]
                new_id = url_to_newid.get(url)
                if new_id:
                    return f"[{new_id}]"
            return ""
        return re.sub(r"\[(\d+)\]", repl, text)

    sections = [(h, remap_citations(p)) for (h, p) in sections_raw]

    # 9) conclusion (grounded)
    conclusion = _llm_write_conclusion(user_query, sections) if sections else None
    if conclusion:
        conclusion = remap_citations(conclusion)

    # 10) final render (NO references in text)
    final_answer = _render_answer_no_refs_in_text(user_query, sections, conclusion)

    # BEST-EFFORT: ensure at least MIN_PARAS paragraphs when possible
    # (If evidence is low, we can't force it.)
    mode = "llm_structure_rewrite_validate_heading_conclusion_grounded_no_refs_in_text"

    return {
        "final_answer": final_answer,
        "references": used_urls_in_order,
        "mode": mode,
    }
