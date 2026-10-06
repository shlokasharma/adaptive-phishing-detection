"""
Tests for the Phase 3 common agent architecture.
"""

from phishing_detection.agents.base import (
    AgentContext,
    AgentExecutionMetadata,
    BaseAgent,
)


class DummyAgent(BaseAgent):
    """Minimal concrete agent used for interface testing."""

    def get_agent_name(self) -> str:
        return "dummy_agent"

    def get_modality(self) -> str:
        return "test"

    def get_capabilities(self) -> list[str]:
        return ["testing"]

    def analyze(self, context: AgentContext) -> dict:
        self.validate_context(context)

        return {
            "agent": self.get_agent_name(),
            "modality": self.get_modality(),
            "input": context.input_text,
        }


def test_agent_context_defaults() -> None:
    context = AgentContext()

    assert context.input_text == ""
    assert context.url is None
    assert context.sender is None
    assert context.headers == {}
    assert context.metadata == {}


def test_agent_context_stores_input() -> None:
    context = AgentContext(
        input_text="Suspicious email",
        url="http://example.com",
        sender="attacker@example.com",
        headers={"subject": "Urgent"},
        metadata={"source": "test"},
    )

    assert context.input_text == "Suspicious email"
    assert context.url == "http://example.com"
    assert context.sender == "attacker@example.com"
    assert context.headers["subject"] == "Urgent"
    assert context.metadata["source"] == "test"


def test_dummy_agent_interface() -> None:
    agent = DummyAgent()

    assert agent.get_agent_name() == "dummy_agent"
    assert agent.get_modality() == "test"
    assert agent.get_capabilities() == ["testing"]


def test_dummy_agent_analysis() -> None:
    agent = DummyAgent()

    context = AgentContext(
        input_text="Test phishing email"
    )

    result = agent.analyze(context)

    assert result["agent"] == "dummy_agent"
    assert result["modality"] == "test"
    assert result["input"] == "Test phishing email"


def test_agent_context_validation() -> None:
    agent = DummyAgent()

    try:
        agent.validate_context("invalid")
    except TypeError as exc:
        assert "AgentContext" in str(exc)
    else:
        raise AssertionError(
            "Expected TypeError for invalid agent context."
        )


def test_agent_can_analyze_defaults_to_true() -> None:
    agent = DummyAgent()

    context = AgentContext(
        input_text="Any input"
    )

    assert agent.can_analyze(context) is True


def test_execution_metadata() -> None:
    agent = DummyAgent()

    metadata = agent.get_execution_metadata(
        execution_time_ms=12.5,
        model_name="test_model",
        additional_metadata={"batch_size": 1},
    )

    assert isinstance(metadata, AgentExecutionMetadata)
    assert metadata.execution_time_ms == 12.5
    assert metadata.model_name == "test_model"
    assert metadata.additional_metadata["batch_size"] == 1
    assert metadata.timestamp