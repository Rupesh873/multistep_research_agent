#main_agent.py
def planning_agent(state):
    user_query = state["user_query"]

    subagents = [
        "web_search",
        "content_reader",
        "news_trend",
        "data_statistics",
        "community_insight",
        "authority_source",
        "verification"
    ]

    plan = {
        "user_query": user_query,
        "subagents": subagents,
        "strategy": "run_all"
    }

    return {
        "plan": plan
    }


