# agents/data_statistics.py
import requests

def data_statistics_node(state):
    queries = state.get("expanded_queries", [])

    for q in queries:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ', '_')}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "extract" in data:
                    return {
                        "data_statistics": {
                            "title": data.get("title"),
                            "summary": data.get("extract")
                        }
                    }
        except:
            continue

    return {
        "data_statistics": {
            "error": "No statistics found"
        }
    }

