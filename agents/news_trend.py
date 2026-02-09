# agents/news_trend.py
from ddgs import DDGS

def news_trend_node(state):
    queries = state.get("expanded_queries", [])
    articles = {}
    
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.news(q, max_results=15):
                    title = r.get("title")
                    url = r.get("url")
                    if title and url:
                        articles[url] = title
    except Exception as e:
        print("[News Trend]", e)

    return {
        "news_trend": [
            f"{title}: {url}" for url, title in articles.items()
        ]
    }

