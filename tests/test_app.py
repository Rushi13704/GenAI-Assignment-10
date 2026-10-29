
import pytest

from app import research_workflow, summarize_results


class FakeLLM:
    def invoke(self, prompt):
        class Response:
            content = (
                "- Finding one\n"
                "- Finding two\n"
                "- Finding three"
            )

        return Response()


def test_research_workflow_returns_three_points():
    result = research_workflow(
        "Renewable energy",
        search_fn=lambda topic: "Sample search results",
        llm=FakeLLM(),
    )

    bullets = [
        line for line in result.splitlines()
        if line.strip().startswith("-")
    ]

    assert len(bullets) == 3


def test_empty_topic_is_rejected():
    with pytest.raises(ValueError):
        research_workflow(
            "   ",
            search_fn=lambda topic: "Sample results",
            llm=FakeLLM(),
        )


def test_empty_search_results_are_rejected():
    with pytest.raises(RuntimeError):
        research_workflow(
            "Artificial intelligence",
            search_fn=lambda topic: "",
            llm=FakeLLM(),
        )
