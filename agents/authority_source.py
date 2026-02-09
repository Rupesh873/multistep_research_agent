# agents/authority_source.py
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

ARXIV_API = "https://export.arxiv.org/api/query"


def authority_source_node(state):
    queries = state.get("expanded_queries", [])
    papers = []
    seen = set()

    for q in queries:
        q_clean = (q or "").strip()
        if not q_clean:
            continue

        # Encode query safely
        search = quote_plus(f"all:{q_clean}")
        url = f"{ARXIV_API}?search_query={search}&start=0&max_results=10"

        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue

            root = ET.fromstring(r.content)

            for e in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title_el = e.find("{http://www.w3.org/2005/Atom}title")
                summary_el = e.find("{http://www.w3.org/2005/Atom}summary")
                id_el = e.find("{http://www.w3.org/2005/Atom}id")

                if title_el is None or summary_el is None or id_el is None:
                    continue

                title = (title_el.text or "").strip()
                abstract = (summary_el.text or "").strip()
                paper_url = (id_el.text or "").strip()

                if not title or not paper_url:
                    continue

                key = (title.lower(), paper_url.lower())
                if key in seen:
                    continue
                seen.add(key)

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "url": paper_url,
                    "source": "arxiv"
                })

        except Exception as e:
            print("[Authority Source]", e)

    return {"authority_source": papers}


