
import logging
import os

from langgraph.graph import StateGraph, START, END

from src.models.state import ContentState
from src.nodes import generate_content, evaluate_content, deliver_result

logger = logging.getLogger(__name__)

MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))



def route_after_evaluation(state: ContentState) -> str:
    
    if state.is_approved:
        logger.info(
            "route: APPROVED (score=%.1f) after %d iteration(s)",
            state.evaluation_score or 0.0,
            state.iteration,
        )
        return "deliver_result"

    if state.iteration >= MAX_RETRIES:
        logger.warning(
            "route: max retries (%d) reached — delivering best effort content "
            "(score=%.1f)",
            MAX_RETRIES,
            state.evaluation_score or 0.0,
        )
        return "deliver_result"          

    logger.info(
        "route: NOT approved (score=%.1f) — regenerating (attempt %d/%d)",
        state.evaluation_score or 0.0,
        state.iteration,
        MAX_RETRIES,
    )
    return "generate_content"



def build_graph() -> StateGraph:
    """Build and return the compiled content-generation graph."""

    builder = StateGraph(ContentState)

    builder.add_node("generate_content", generate_content)
    builder.add_node("evaluate_content", evaluate_content)
    builder.add_node("deliver_result", deliver_result)

    builder.add_edge(START, "generate_content")
    builder.add_edge("generate_content", "evaluate_content")

    builder.add_conditional_edges(
        "evaluate_content",
        route_after_evaluation,
        {
            "deliver_result": "deliver_result",
            "generate_content": "generate_content",
        },
    )

    builder.add_edge("deliver_result", END)

    return builder.compile()

content_graph = build_graph()
