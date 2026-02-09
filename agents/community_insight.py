# agents/community_insight.py
from typing import List, Dict
import requests


def community_insight_node(state):
    queries = state.get("expanded_queries", [])
    insights: List[Dict] = []
    seen = set()

    url = "https://hn.algolia.com/api/v1/search"

    for q in queries:
        params = {
            "query": q,
            "tags": "story",
            "hitsPerPage": 30
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            for hit in data.get("hits", []):
                title = hit.get("title")
                link = hit.get("url")

                if not title or title in seen:
                    continue

                insights.append({
                    "query_used": q,
                    "title": title,
                    "url": link,
                    "points": hit.get("points", 0),
                    "comments": hit.get("num_comments", 0)
                })

                seen.add(title)

        except Exception as e:
            print("[Community Insight]", e)

    return {
        "community_insight": insights
    }

