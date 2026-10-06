"""
Tests for the Phase 3 structured evidence schema.
"""

import pytest

from phishing_detection.agents.evidence import Evidence


def test_basic_evidence_creation() -> None:
    evidence = Evidence(
        evidence_type="url",
        source="url_analysis_agent",
        description="URL was classified as phishing.",
        value="http://example.com",
        polarity="supports_phishing",
        confidence=0.95,
        reliability=0.90,
    )

    assert evidence.evidence_type == "url"
    assert evidence.source == "url_analysis_agent"
    assert evidence.description == (
        "URL was classified as phishing."
    )
    assert evidence.value == "http://example.com"
    assert evidence.polarity == "supports_phishing"
    assert evidence.confidence == 0.95
    assert evidence.reliability == 0.90


def test_evidence_to_dict() -> None:
    evidence = Evidence(
        evidence_type="email_text",
        source="email_analysis_agent",
        description="Suspicious language detected.",
        polarity="supports_phishing",
        confidence=0.88,
        reliability=0.92,
        metadata={"feature": "urgency"},
    )

    result = evidence.to_dict()

    assert isinstance(result, dict)
    assert result["evidence_type"] == "email_text"
    assert result["source"] == "email_analysis_agent"
    assert result["polarity"] == "supports_phishing"
    assert result["confidence"] == 0.88
    assert result["reliability"] == 0.92
    assert result["metadata"]["feature"] == "urgency"


def test_phishing_polarity_property() -> None:
    evidence = Evidence(
        evidence_type="url",
        source="test_agent",
        description="Suspicious URL.",
        polarity="supports_phishing",
    )

    assert evidence.supports_phishing is True
    assert evidence.supports_legitimate is False
    assert evidence.is_neutral is False


def test_legitimate_polarity_property() -> None:
    evidence = Evidence(
        evidence_type="sender",
        source="test_agent",
        description="Sender domain is trusted.",
        polarity="supports_legitimate",
    )

    assert evidence.supports_phishing is False
    assert evidence.supports_legitimate is True
    assert evidence.is_neutral is False


def test_neutral_polarity_property() -> None:
    evidence = Evidence(
        evidence_type="header",
        source="test_agent",
        description="No decisive signal.",
    )

    assert evidence.supports_phishing is False
    assert evidence.supports_legitimate is False
    assert evidence.is_neutral is True


def test_invalid_polarity_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid evidence polarity"):
        Evidence(
            evidence_type="url",
            source="test_agent",
            description="Invalid evidence.",
            polarity="unknown",
        )


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Evidence(
            evidence_type="url",
            source="test_agent",
            description="Invalid confidence.",
            confidence=1.5,
        )


def test_invalid_reliability_is_rejected() -> None:
    with pytest.raises(ValueError, match="reliability"):
        Evidence(
            evidence_type="url",
            source="test_agent",
            description="Invalid reliability.",
            reliability=-0.1,
        )


def test_empty_evidence_type_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="evidence_type must not be empty",
    ):
        Evidence(
            evidence_type="",
            source="test_agent",
            description="Some evidence.",
        )


def test_empty_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        Evidence(
            evidence_type="url",
            source="",
            description="Some evidence.",
        )


def test_empty_description_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="description must not be empty",
    ):
        Evidence(
            evidence_type="url",
            source="test_agent",
            description="",
        )


def test_metadata_must_be_dictionary() -> None:
    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary",
    ):
        Evidence(
            evidence_type="url",
            source="test_agent",
            description="Some evidence.",
            metadata="invalid",  # type: ignore[arg-type]
        )