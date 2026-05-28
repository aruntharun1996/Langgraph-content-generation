import logging

from src.models.state import ContentState

logger = logging.getLogger(__name__)


def deliver_result(state: ContentState) -> ContentState:
    """LangGraph node: package the approved content as the final result."""

    logger.info(
        "deliver_result node | total_iterations=%d | score=%.1f",
        state.iteration,
        state.evaluation_score or 0.0,
    )

    return state.model_copy(
        update={
            "final_content": state.generated_content,
            "error": None,
        }
    )
