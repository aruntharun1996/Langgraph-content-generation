
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.models.state import ContentState
from src.utils.llm_client import get_evaluation_llm

logger = logging.getLogger(__name__)

APPROVAL_THRESHOLD = 7.0  

_SYSTEM_PROMPT = """You are a strict content quality evaluator.
Assess the provided content against the original user request.

Respond ONLY with a valid JSON object — nothing else:
{
  "score": <float 0.0-10.0>,
  "is_approved": <true|false>,
  "feedback": "<actionable improvement points, or 'Content meets quality standards.' if approved>"
}

Scoring rubric:
  9-10 : Exceptional — exceeds expectations, publish-ready.
  7-8  : Good        — meets all requirements with minor polish needed.
  5-6  : Adequate    — partially meets requirements; clear gaps remain.
  3-4  : Poor        — major issues with structure, accuracy, or relevance.
  0-2  : Unacceptable— off-topic, incoherent, or harmful content.

Set is_approved to true ONLY when score >= 7.0."""


def _build_eval_prompt(user_request: str, content: str, iteration: int) -> str:
    return f"""Evaluate the following content (iteration {iteration}) against the user request.

USER REQUEST:
{user_request}

CONTENT TO EVALUATE:
{content}

Return your evaluation as a JSON object following the schema provided."""


def _parse_evaluation(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in evaluator response: {raw!r}")


def evaluate_content(state: ContentState) -> ContentState:
    """LangGraph node: evaluate the generated content with Gemini."""

    if not state.generated_content:
        logger.warning("evaluate_content: no content to evaluate — marking as failed")
        return state.model_copy(
            update={
                "is_approved": False,
                "evaluation_feedback": "No content was generated.",
                "evaluation_score": 0.0,
            }
        )

    logger.info(
        "evaluate_content node | iteration=%d | content_length=%d",
        state.iteration,
        len(state.generated_content),
    )

    llm = get_evaluation_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_eval_prompt(
                user_request=state.user_request,
                content=state.generated_content,
                iteration=state.iteration,
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        parsed = _parse_evaluation(response.content)

        score: float = float(parsed.get("score", 0.0))
        is_approved: bool = bool(parsed.get("is_approved", score >= APPROVAL_THRESHOLD))
        feedback: str = parsed.get("feedback", "No specific feedback provided.")

        logger.info(
            "evaluate_content: score=%.1f | approved=%s | feedback='%s...'",
            score,
            is_approved,
            feedback[:80],
        )

        return state.model_copy(
            update={
                "is_approved": is_approved,
                "evaluation_score": score,
                "evaluation_feedback": feedback,
                "error": None,
            }
        )

    except Exception as exc:
        logger.exception("evaluate_content: evaluation failed — %s", exc)
        return state.model_copy(
            update={
                "is_approved": False,
                "evaluation_score": 0.0,
                "evaluation_feedback": "Evaluation failed; please regenerate.",
                "error": str(exc),
            }
        )
