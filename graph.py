# graph.py
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from getting_multiple_queries import generate_multiple_queries
from main_agent import planning_agent
from aggregation_scoring import aggregation_node
from repair_dedup import repair_and_dedup_node
from scrape_sources import scrape_sources_node
from content_processing import content_processing_node
from vector_index import vector_index_node

from final_synthesis import final_synthesis_node  # ✅ STEP 10

from agents.web_search import web_search_node
from agents.news_trend import news_trend_node
from agents.data_statistics import data_statistics_node
from agents.community_insight import community_insight_node
from agents.authority_source import authority_source_node
from agents.content_reader import content_reader_node
from agents.verification import verification_node


class MSRAState(TypedDict, total=False):
    # Step 1–3
    user_query: str
    expanded_queries: List[str]
    plan: Dict[str, Any]

    # Step 4
    web_search: List[str]
    news_trend: List[str]
    data_statistics: Dict
    community_insight: List[Dict]
    authority_source: List[Dict]
    content_reader: Dict[str, str]
    verification: Dict

    # Step 5–6
    evidence: List[Dict]
    clean_evidence: List[Dict]
    repair_stats: Dict

    # Step 7–8
    scraped_content: List[Dict]
    processed_chunks: List[Dict]

    # Step 9
    vector_index: List[Dict]

    # Step 10
    final_answer: str
    references: List[str]
    mode: str


def build_graph():
    graph = StateGraph(MSRAState)

    # Step 2–3
    graph.add_node("query_expansion", generate_multiple_queries)
    graph.add_node("planning", planning_agent)

    # Step 4
    graph.add_node("web_search", web_search_node)
    graph.add_node("news_trend", news_trend_node)
    graph.add_node("data_statistics", data_statistics_node)
    graph.add_node("community_insight", community_insight_node)
    graph.add_node("authority_source", authority_source_node)
    graph.add_node("content_reader", content_reader_node)
    graph.add_node("verification", verification_node)

    # Step 5–6
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("repair_dedup", repair_and_dedup_node)

    # Step 7–9
    graph.add_node("scrape_sources", scrape_sources_node)
    graph.add_node("content_processing", content_processing_node)
    graph.add_node("vector_indexing", vector_index_node)

    # Step 10 ✅
    graph.add_node("final_synthesis", final_synthesis_node)

    # -------- Flow --------
    graph.set_entry_point("query_expansion")
    graph.add_edge("query_expansion", "planning")

    graph.add_edge("planning", "web_search")
    graph.add_edge("planning", "news_trend")
    graph.add_edge("planning", "data_statistics")
    graph.add_edge("planning", "community_insight")
    graph.add_edge("planning", "authority_source")

    graph.add_edge("web_search", "content_reader")

    graph.add_edge("content_reader", "verification")
    graph.add_edge("news_trend", "verification")
    graph.add_edge("data_statistics", "verification")
    graph.add_edge("community_insight", "verification")
    graph.add_edge("authority_source", "verification")

    graph.add_edge("verification", "aggregation")
    graph.add_edge("aggregation", "repair_dedup")
    graph.add_edge("repair_dedup", "scrape_sources")
    graph.add_edge("scrape_sources", "content_processing")
    graph.add_edge("content_processing", "vector_indexing")

    # Step 10
    graph.add_edge("vector_indexing", "final_synthesis")
    graph.add_edge("final_synthesis", END)

    return graph.compile()