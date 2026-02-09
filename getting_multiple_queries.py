#getting_multiple_queries.py
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv("multistep_research_agent.env")


def generate_multiple_queries(state):
    user_query = state["user_query"]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    system_prompt = (
        "Generate 7 concise search queries.\n"
        "Include the original query.\n"
        "Return STRICT JSON:\n"
        '{ "queries": ["q1", "q2"] }'
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    response = llm.invoke(messages)

    try:
        data = json.loads(response.content)
        queries = data.get("queries", [])
    except Exception:
        queries = [user_query]

    cleaned = []
    seen = set()

    for q in queries:
        if isinstance(q, str):
            q = q.strip()
            if q and q.lower() not in seen:
                cleaned.append(q)
                seen.add(q.lower())

    if user_query not in cleaned:
        cleaned.insert(0, user_query)

    return {
        "expanded_queries": cleaned
    }
