
import logging
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.state import ContentState
from src.utils.llm_client import get_generation_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert content writer.
Your job is to create high-quality, engaging, and accurate content.

Guidelines:
- Be clear, concise, and compelling.
- Match the tone and format to the request (blog post, email, tweet, etc.).
- Ensure factual accuracy and logical structure.
- Use proper grammar and punctuation.
- Deliver ONLY the final content — no meta-commentary, no preamble."""


def _build_initial_prompt(user_request: str) -> str:
    return f"""Please create content based on the following request:

USER REQUEST:
{user_request}

Produce well-structured, high-quality content that fully satisfies the request."""


def _build_regeneration_prompt(
    user_request: str,
    previous_content: str,
    feedback: str,
    iteration: int,
) -> str:
    return f"""You are improving a content draft that did not pass quality review.

ORIGINAL USER REQUEST:
{user_request}

PREVIOUS CONTENT (iteration {iteration}):
{previous_content}

EVALUATOR FEEDBACK — THINGS TO IMPROVE:
{feedback}

Please rewrite the content, addressing every point in the feedback while
still fulfilling the original request. Deliver only the improved content."""


def generate_content(state: ContentState) -> ContentState:
    """LangGraph node: generate or regenerate content using Gemini."""

    logger.info(
        "generate_content node | iteration=%d | request='%s...'",
        state.iteration + 1,
        state.user_request[:60],
    )

    llm = get_generation_llm()

    # Pick the right prompt
    if state.iteration == 0 or state.generated_content is None:
        user_prompt = _build_initial_prompt(state.user_request)
    else:
        user_prompt = _build_regeneration_prompt(
            user_request=state.user_request,
            previous_content=state.generated_content,
            feedback=state.evaluation_feedback or "General quality improvements needed.",
            iteration=state.iteration,
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        new_content = response.content.strip()

        updated_previous = list(state.previous_contents)
        if state.generated_content:
            updated_previous.append(state.generated_content)

        updated_feedbacks = list(state.previous_feedbacks)
        if state.evaluation_feedback:
            updated_feedbacks.append(state.evaluation_feedback)

        logger.info("generate_content: produced %d characters", len(new_content))

        return state.model_copy(
            update={
                "generated_content": new_content,
                "iteration": state.iteration + 1,
                "previous_contents": updated_previous,
                "previous_feedbacks": updated_feedbacks,
                # Reset evaluation fields for the new draft
                "is_approved": None,
                "evaluation_feedback": None,
                "evaluation_score": None,
                "error": None,
            }
        )

    except Exception as exc:
        logger.exception("generate_content: LLM call failed — %s", exc)
        return state.model_copy(update={"error": str(exc)})
