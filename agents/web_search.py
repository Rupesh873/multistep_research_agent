# agents/web_search.py
from ddgs import DDGS

def web_search_node(state):
    queries = state.get("expanded_queries", [])
    all_urls = set()

    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=20):
                    if "href" in r:
                        all_urls.add(r["href"])
    except Exception as e:
        print("[Web Search]", e)

    return {
        "web_search": list(all_urls)
    }
