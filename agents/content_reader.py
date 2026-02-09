#content_reader.py
import requests
from bs4 import BeautifulSoup

def content_reader_node(state):
    urls = state.get("web_search", [])
    contents = {}

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls[:5]:  # limit for demo
        try:
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(["script", "style"]):
                s.decompose()
            contents[url] = soup.get_text(separator=" ")[:6000]
        except:
            contents[url] = "Failed"

    return {"content_reader": contents}
