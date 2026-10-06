from __future__ import annotations

from phishing_detection.agents.base import AgentContext, BaseAgent
from phishing_detection.agents.result import AgentResult
from phishing_detection.pipeline import PhishingDetectionPipeline


class FakeAgent(BaseAgent):
    """Simple deterministic agent for pipeline testing."""

    def __init__(
        self,
        name: str,
        modality: str,
        probability: float,
        can_run: bool = True,
    ) -> None:
        self.name = name
        self.modality = modality
        self.probability = probability
        self.can_run = can_run

    def get_agent_name(self) -> str:
        return self.name

    def get_modality(self) -> str:
        return self.modality

    def get_capabilities(self) -> list[str]:
        return [f"{self.modality}_analysis"]

    def can_analyze(self, context: AgentContext) -> bool:
        return self.can_run

    def analyze(self, context: AgentContext) -> AgentResult:
        return AgentResult.from_probability(
            agent_name=self.name,
            modality=self.modality,
            phishing_probability=self.probability,
            evidence=[],
        )


def create_pipeline(
    email_probability: float = 0.95,
    url_probability: float = 0.95,
    sender_probability: float = 0.95,
):
    """Create a pipeline with deterministic fake agents."""

    from phishing_detection.orchestration import OrchestratorAgent

    email_agent = FakeAgent(
        name="email_analysis_agent",
        modality="email",
        probability=email_probability,
    )

    url_agent = FakeAgent(
        name="url_analysis_agent",
        modality="url",
        probability=url_probability,
    )

    sender_agent = FakeAgent(
        name="sender_analysis_agent",
        modality="sender",
        probability=sender_probability,
    )

    orchestrator = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        confidence_threshold=0.90,
        max_agents=3,
    )

    return PhishingDetectionPipeline(
        orchestrator=orchestrator,
    )


def test_pipeline_initialization():
    """Pipeline should initialize correctly."""

    pipeline = create_pipeline()

    assert pipeline.pipeline_name == (
        "adaptive_phishing_detection_pipeline"
    )

    assert pipeline.pipeline_version == "0.1.0"


def test_pipeline_rejects_missing_orchestrator():
    """Pipeline should reject a missing orchestrator."""

    try:
        PhishingDetectionPipeline(orchestrator=None)
    except ValueError as exc:
        assert "orchestrator" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing orchestrator."
        )


def test_build_context():
    """Pipeline should convert inputs into AgentContext."""

    pipeline = create_pipeline()

    context = pipeline.build_context(
        email_text="  test email  ",
        url="  https://example.com  ",
        sender="  sender@example.com  ",
        headers={"From": "sender@example.com"},
        metadata={"custom_key": "custom_value"},
    )

    assert context.input_text == "test email"
    assert context.url == "https://example.com"
    assert context.sender == "sender@example.com"

    assert context.headers["From"] == "sender@example.com"
    assert context.metadata["custom_key"] == "custom_value"

    assert (
        context.metadata["pipeline_name"]
        == "adaptive_phishing_detection_pipeline"
    )


def test_pipeline_email_only_high_confidence():
    """High-confidence email analysis should terminate early."""

    pipeline = create_pipeline(
        email_probability=0.98,
        url_probability=0.98,
        sender_probability=0.98,
    )

    result = pipeline.analyze(
        email_text="This is a phishing email.",
    )

    assert result.is_phishing
    assert result.agent_count == 1
    assert result.evidence_count == 0
    assert result.confidence >= 0.90


def test_pipeline_adaptive_email_url_path():
    """Uncertain email should trigger URL analysis."""

    pipeline = create_pipeline(
        email_probability=0.60,
        url_probability=0.95,
        sender_probability=0.95,
    )

    result = pipeline.analyze(
        email_text="Suspicious email.",
        url="https://example.com",
    )

    assert result.agent_count == 2

    executed_agents = [
        agent.agent_name
        for agent in result.agent_results
    ]

    assert "email_analysis_agent" in executed_agents
    assert "url_analysis_agent" in executed_agents


def test_pipeline_adaptive_three_agent_path():
    """Uncertain email and URL should trigger sender analysis."""

    pipeline = create_pipeline(
        email_probability=0.55,
        url_probability=0.55,
        sender_probability=0.95,
    )

    result = pipeline.analyze(
        email_text="Suspicious email.",
        url="https://example.com",
        sender="sender@example.com",
    )

    assert result.agent_count == 3

    executed_agents = [
        agent.agent_name
        for agent in result.agent_results
    ]

    assert "email_analysis_agent" in executed_agents
    assert "url_analysis_agent" in executed_agents
    assert "sender_analysis_agent" in executed_agents


def test_pipeline_handles_email_only_input():
    """Email-only input should still produce a final decision."""

    pipeline = create_pipeline(
        email_probability=0.10,
    )

    result = pipeline.analyze(
        email_text="This appears legitimate.",
    )

    assert result.is_legitimate
    assert result.label == 0
    assert 0.0 <= result.phishing_probability <= 1.0


def test_pipeline_handles_url_only_input():
    """URL-only input should produce a valid result."""

    pipeline = create_pipeline(
        email_probability=0.50,
        url_probability=0.95,
        sender_probability=0.95,
    )

    result = pipeline.analyze(
        url="https://example.com",
    )

    assert result.label in (0, 1)
    assert 0.0 <= result.phishing_probability <= 1.0


def test_pipeline_handles_sender_only_input():
    """Sender-only input should produce a valid result."""

    pipeline = create_pipeline(
        email_probability=0.50,
        url_probability=0.50,
        sender_probability=0.90,
    )

    result = pipeline.analyze(
        sender="sender@example.com",
    )

    assert result.label in (0, 1)
    assert 0.0 <= result.phishing_probability <= 1.0


def test_pipeline_generates_decision_trace():
    """Pipeline should expose the orchestrator decision trace."""

    pipeline = create_pipeline(
        email_probability=0.55,
        url_probability=0.95,
    )

    result = pipeline.analyze(
        email_text="Suspicious email.",
        url="https://example.com",
    )

    assert len(result.decision_trace) >= 1

    first_step = result.decision_trace[0]

    assert first_step.agent_name == "email_analysis_agent"


def test_pipeline_serialization():
    """Pipeline result should be serializable."""

    pipeline = create_pipeline(
        email_probability=0.95,
    )

    result = pipeline.analyze(
        email_text="Test email.",
    )

    serialized = result.to_dict()

    assert isinstance(serialized, dict)

    assert "label" in serialized
    assert "label_name" in serialized
    assert "phishing_probability" in serialized
    assert "confidence" in serialized
    assert "uncertainty" in serialized
    assert "agent_results" in serialized
    assert "decision_trace" in serialized
    assert "evidence" in serialized
    assert "pipeline_metadata" in serialized


def test_pipeline_metadata():
    """Pipeline should record execution metadata."""

    pipeline = create_pipeline()

    result = pipeline.analyze(
        email_text="Test email.",
    )

    metadata = result.pipeline_metadata

    assert metadata["pipeline_name"] == (
        "adaptive_phishing_detection_pipeline"
    )

    assert metadata["pipeline_version"] == "0.1.0"

    assert "execution_timestamp" in metadata
    assert "execution_time_ms" in metadata
    assert "input_modalities" in metadata
    assert "agent_count" in metadata
    assert "evidence_count" in metadata


def test_pipeline_input_modalities():
    """Pipeline metadata should correctly describe supplied inputs."""

    pipeline = create_pipeline(
        email_probability=0.98,
    )

    result = pipeline.analyze(
        email_text="Test email.",
        url="https://example.com",
        sender="sender@example.com",
        headers={"From": "sender@example.com"},
    )

    modalities = result.pipeline_metadata["input_modalities"]

    assert modalities["email"] is True
    assert modalities["url"] is True
    assert modalities["sender"] is True
    assert modalities["headers"] is True


def test_pipeline_preserves_custom_metadata():
    """
    Pipeline should preserve custom context metadata without
    interfering with pipeline metadata.
    """

    pipeline = create_pipeline(
        email_probability=0.98,
    )

    context = pipeline.build_context(
        email_text="Test email.",
        metadata={
            "request_id": "test-001",
            "experiment": "phase3.9",
        },
    )

    assert context.metadata["request_id"] == "test-001"
    assert context.metadata["experiment"] == "phase3.9"


def test_pipeline_result_properties():
    """PipelineResult convenience properties should work."""

    pipeline = create_pipeline(
        email_probability=0.95,
    )

    result = pipeline.analyze(
        email_text="Test email.",
    )

    assert result.is_phishing is True
    assert result.is_legitimate is False
    assert result.agent_count == 1
    assert result.evidence_count == 0