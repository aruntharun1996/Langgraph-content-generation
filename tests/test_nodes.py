"""
tests/test_nodes.py
────────────────────
Unit tests for all three nodes using mocked LLM calls.
Run with:  pytest tests/ -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models.state import ContentState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = text
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# generate_content tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateContent:

    @patch("src.nodes.generate_content.get_generation_llm")
    def test_initial_generation(self, mock_get_llm):
        from src.nodes.generate_content import generate_content

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("This is great content.")
        mock_get_llm.return_value = mock_llm

        state = ContentState(user_request="Write a blog post about Python.")
        result = generate_content(state)

        assert result.generated_content == "This is great content."
        assert result.iteration == 1
        assert result.is_approved is None
        assert result.error is None

    @patch("src.nodes.generate_content.get_generation_llm")
    def test_regeneration_uses_feedback(self, mock_get_llm):
        from src.nodes.generate_content import generate_content

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("Improved content.")
        mock_get_llm.return_value = mock_llm

        state = ContentState(
            user_request="Write a blog post about Python.",
            generated_content="Old content.",
            iteration=1,
            is_approved=False,
            evaluation_feedback="Needs more depth.",
        )
        result = generate_content(state)

        assert result.generated_content == "Improved content."
        assert result.iteration == 2
        assert "Old content." in result.previous_contents
        # Ensure feedback made it into the prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert any("Needs more depth" in str(m.content) for m in call_args)

    @patch("src.nodes.generate_content.get_generation_llm")
    def test_llm_error_captured(self, mock_get_llm):
        from src.nodes.generate_content import generate_content

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        mock_get_llm.return_value = mock_llm

        state = ContentState(user_request="Write something.")
        result = generate_content(state)

        assert result.error == "API timeout"


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_content tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateContent:

    @patch("src.nodes.evaluate_content.get_evaluation_llm")
    def test_approved_content(self, mock_get_llm):
        from src.nodes.evaluate_content import evaluate_content

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(
            '{"score": 8.5, "is_approved": true, "feedback": "Content meets quality standards."}'
        )
        mock_get_llm.return_value = mock_llm

        state = ContentState(
            user_request="Write a post.",
            generated_content="Excellent content here.",
            iteration=1,
        )
        result = evaluate_content(state)

        assert result.is_approved is True
        assert result.evaluation_score == 8.5

    @patch("src.nodes.evaluate_content.get_evaluation_llm")
    def test_rejected_content(self, mock_get_llm):
        from src.nodes.evaluate_content import evaluate_content

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(
            '{"score": 4.0, "is_approved": false, "feedback": "Too short and lacks examples."}'
        )
        mock_get_llm.return_value = mock_llm

        state = ContentState(
            user_request="Write a post.",
            generated_content="Short.",
            iteration=1,
        )
        result = evaluate_content(state)

        assert result.is_approved is False
        assert result.evaluation_score == 4.0
        assert "Too short" in result.evaluation_feedback

    @patch("src.nodes.evaluate_content.get_evaluation_llm")
    def test_no_content_returns_failed(self, mock_get_llm):
        from src.nodes.evaluate_content import evaluate_content

        state = ContentState(user_request="Write a post.")
        result = evaluate_content(state)

        assert result.is_approved is False
        assert result.evaluation_score == 0.0

    @patch("src.nodes.evaluate_content.get_evaluation_llm")
    def test_json_with_markdown_fences(self, mock_get_llm):
        from src.nodes.evaluate_content import evaluate_content

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(
            '```json\n{"score": 7.5, "is_approved": true, "feedback": "Good."}\n```'
        )
        mock_get_llm.return_value = mock_llm

        state = ContentState(
            user_request="Write.",
            generated_content="Some content.",
            iteration=1,
        )
        result = evaluate_content(state)

        assert result.is_approved is True
        assert result.evaluation_score == 7.5


# ─────────────────────────────────────────────────────────────────────────────
# deliver_result tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeliverResult:

    def test_promotes_to_final(self):
        from src.nodes.deliver_result import deliver_result

        state = ContentState(
            user_request="Write.",
            generated_content="Final approved content.",
            is_approved=True,
            evaluation_score=8.0,
            iteration=2,
        )
        result = deliver_result(state)

        assert result.final_content == "Final approved content."
        assert result.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Graph routing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouting:

    def test_route_approved(self):
        from src.graph.content_graph import route_after_evaluation

        state = ContentState(
            user_request="Write.",
            is_approved=True,
            evaluation_score=8.0,
            iteration=1,
        )
        assert route_after_evaluation(state) == "deliver_result"

    def test_route_not_approved_retry(self):
        from src.graph.content_graph import route_after_evaluation

        state = ContentState(
            user_request="Write.",
            is_approved=False,
            evaluation_score=5.0,
            iteration=1,
        )
        assert route_after_evaluation(state) == "generate_content"

    def test_route_max_retries_delivers(self):
        from src.graph.content_graph import route_after_evaluation

        state = ContentState(
            user_request="Write.",
            is_approved=False,
            evaluation_score=4.0,
            iteration=3,   # equals MAX_RETRIES default
        )
        assert route_after_evaluation(state) == "deliver_result"
