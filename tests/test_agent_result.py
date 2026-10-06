"""
Tests for the Phase 3 standardized AgentResult schema.
"""

import pytest

from phishing_detection.agents.base import AgentExecutionMetadata
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult


def make_evidence() -> Evidence:
    """Create reusable test evidence."""

    return Evidence(
        evidence_type="url",
        source="url_analysis_agent",
        description="Suspicious URL detected.",
        value="http://example.com",
        polarity="supports_phishing",
        confidence=0.95,
        reliability=0.90,
    )


def test_basic_agent_result_creation() -> None:
    result = AgentResult(
        agent_name="url_analysis_agent",
        modality="url",
        label=1,
        label_name="phishing",
        phishing_probability=0.95,
        confidence=0.95,
        uncertainty=0.05,
        evidence=[make_evidence()],
    )

    assert result.agent_name == "url_analysis_agent"
    assert result.modality == "url"
    assert result.label == 1
    assert result.label_name == "phishing"
    assert result.phishing_probability == 0.95
    assert result.confidence == 0.95
    assert result.uncertainty == 0.05
    assert result.evidence_count == 1


def test_legitimate_result() -> None:
    result = AgentResult(
        agent_name="email_analysis_agent",
        modality="email",
        label=0,
        label_name="legitimate",
        phishing_probability=0.10,
        confidence=0.90,
        uncertainty=0.10,
    )

    assert result.is_legitimate is True
    assert result.is_phishing is False


def test_phishing_result() -> None:
    result = AgentResult(
        agent_name="email_analysis_agent",
        modality="email",
        label=1,
        label_name="phishing",
        phishing_probability=0.90,
        confidence=0.90,
        uncertainty=0.10,
    )

    assert result.is_phishing is True
    assert result.is_legitimate is False


def test_result_to_dict() -> None:
    result = AgentResult(
        agent_name="url_analysis_agent",
        modality="url",
        label=1,
        label_name="phishing",
        phishing_probability=0.90,
        confidence=0.90,
        uncertainty=0.10,
        evidence=[make_evidence()],
        metadata={"test": True},
    )

    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["agent_name"] == "url_analysis_agent"
    assert data["modality"] == "url"
    assert data["label"] == 1
    assert data["label_name"] == "phishing"
    assert data["phishing_probability"] == 0.90
    assert data["confidence"] == 0.90
    assert data["uncertainty"] == 0.10
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["evidence_type"] == "url"
    assert data["metadata"]["test"] is True


def test_evidence_count() -> None:
    result = AgentResult(
        agent_name="email_analysis_agent",
        modality="email",
        label=1,
        label_name="phishing",
        phishing_probability=0.80,
        confidence=0.80,
        uncertainty=0.20,
        evidence=[
            make_evidence(),
            Evidence(
                evidence_type="email_text",
                source="email_analysis_agent",
                description="Credential request detected.",
                polarity="supports_phishing",
            ),
        ],
    )

    assert result.evidence_count == 2


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid label"):
        AgentResult(
            agent_name="test_agent",
            modality="test",
            label=2,
            label_name="phishing",
            phishing_probability=0.80,
            confidence=0.80,
            uncertainty=0.20,
        )


def test_label_name_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="label_name does not match label",
    ):
        AgentResult(
            agent_name="test_agent",
            modality="test",
            label=1,
            label_name="legitimate",
            phishing_probability=0.80,
            confidence=0.80,
            uncertainty=0.20,
        )


def test_invalid_phishing_probability_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="phishing_probability",
    ):
        AgentResult(
            agent_name="test_agent",
            modality="test",
            label=1,
            label_name="phishing",
            phishing_probability=1.5,
            confidence=0.90,
            uncertainty=0.10,
        )


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="confidence",
    ):
        AgentResult(
            agent_name="test_agent",
            modality="test",
            label=1,
            label_name="phishing",
            phishing_probability=0.90,
            confidence=-0.1,
            uncertainty=1.1,
        )


def test_invalid_uncertainty_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="uncertainty",
    ):
        AgentResult(
            agent_name="test_agent",
            modality="test",
            label=1,
            label_name="phishing",
            phishing_probability=0.90,
            confidence=0.90,
            uncertainty=1.5,
        )


def test_invalid_evidence_item_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="Every item in evidence",
    ):
        AgentResult(
            agent_name="test_agent",
            modality="test",
            label=1,
            label_name="phishing",
            phishing_probability=0.90,
            confidence=0.90,
            uncertainty=0.10,
            evidence=["invalid"],  # type: ignore[list-item]
        )


def test_execution_metadata_is_supported() -> None:
    metadata = AgentExecutionMetadata(
        execution_time_ms=15.2,
        model_name="test_model",
    )

    result = AgentResult(
        agent_name="test_agent",
        modality="email",
        label=1,
        label_name="phishing",
        phishing_probability=0.90,
        confidence=0.90,
        uncertainty=0.10,
        execution_metadata=metadata,
    )

    assert result.execution_metadata is not None
    assert result.execution_metadata.execution_time_ms == 15.2
    assert result.execution_metadata.model_name == "test_model"


def test_from_probability_phishing() -> None:
    result = AgentResult.from_probability(
        agent_name="url_analysis_agent",
        modality="url",
        phishing_probability=0.80,
    )

    assert result.label == 1
    assert result.label_name == "phishing"
    assert result.confidence == 0.80
    assert result.uncertainty == 0.20


def test_from_probability_legitimate() -> None:
    result = AgentResult.from_probability(
        agent_name="url_analysis_agent",
        modality="url",
        phishing_probability=0.20,
    )

    assert result.label == 0
    assert result.label_name == "legitimate"
    assert result.confidence == 0.80
    assert result.uncertainty == 0.20


def test_from_probability_with_evidence() -> None:
    evidence = make_evidence()

    result = AgentResult.from_probability(
        agent_name="url_analysis_agent",
        modality="url",
        phishing_probability=0.92,
        evidence=[evidence],
        metadata={"source": "phase3_test"},
    )

    assert result.is_phishing is True
    assert result.evidence_count == 1
    assert result.evidence[0] is evidence
    assert result.metadata["source"] == "phase3_test"


def test_from_probability_invalid_probability() -> None:
    with pytest.raises(
        ValueError,
        match="phishing_probability",
    ):
        AgentResult.from_probability(
            agent_name="test_agent",
            modality="test",
            phishing_probability=1.2,
        )