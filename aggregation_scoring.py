# aggregation_scoring.py
from typing import Dict, Any, List


def aggregation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 5: Raw Evidence Aggregation

    - Collect raw outputs from subagents
    - NO deduplication
    - NO repair
    - NO synthesis
    """

    evidence: List[Dict[str, Any]] = []
    warnings: List[str] = []

    subagents = [
        "web_search",
        "news_trend",
        "data_statistics",
        "community_insight",
        "authority_source",
        "content_reader",
    ]

    for agent in subagents:
        data = state.get(agent)

        if not data:
            warnings.append(f"No data from {agent}")
            continue

        if isinstance(data, list):
            for item in data:
                evidence.append({
                    "source": agent,
                    "data": item
                })
        else:
            evidence.append({
                "source": agent,
                "data": data
            })

    return {
        # 🔑 MUST match MSRAState
        "evidence": evidence,
        "warnings": warnings
    }