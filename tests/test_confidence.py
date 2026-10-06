from __future__ import annotations

import pytest

from phishing_detection.agents.confidence import (
    ConfidenceAssessment,
    assess_agent_result,
    calculate_aggregate_evidence_strength,
    calculate_confidence,
    calculate_evidence_strength,
    calculate_uncertainty,
)
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult


def create_evidence(
    *,
    confidence: float = 0.8,
    reliability: float = 0.9,
    polarity: str = "supports_phishing",
) -> Evidence:
    return Evidence(
        evidence_type="test",
        source="test_agent",
        description="Test evidence",
        value="test",
        polarity=polarity,
        confidence=confidence,
        reliability=reliability,
    )


def create_result(
    probability: float,
    evidence=None,
) -> AgentResult:
    return AgentResult.from_probability(
        agent_name="test_agent",
        modality="test",
        phishing_probability=probability,
        evidence=evidence or [],
    )


def test_confidence_for_high_phishing_probability():
    assert (
        calculate_confidence(0.90)
        == pytest.approx(0.90)
    )


def test_confidence_for_low_phishing_probability():
    assert (
        calculate_confidence(0.10)
        == pytest.approx(0.90)
    )


def test_confidence_for_balanced_probability():
    assert (
        calculate_confidence(0.50)
        == pytest.approx(0.50)
    )


def test_uncertainty_is_complement():
    assert (
        calculate_uncertainty(0.90)
        == pytest.approx(0.10)
    )


def test_uncertainty_at_zero_confidence():
    assert (
        calculate_uncertainty(0.0)
        == pytest.approx(1.0)
    )


def test_uncertainty_at_full_confidence():
    assert (
        calculate_uncertainty(1.0)
        == pytest.approx(0.0)
    )


def test_invalid_probability_raises():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        calculate_confidence(1.5)


def test_negative_probability_raises():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        calculate_confidence(-0.1)


def test_invalid_confidence_raises():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        calculate_uncertainty(1.2)


def test_evidence_strength():
    evidence = create_evidence(
        confidence=0.8,
        reliability=0.9,
    )

    assert (
        calculate_evidence_strength(evidence)
        == pytest.approx(0.72)
    )


def test_evidence_strength_with_zero_confidence():
    evidence = create_evidence(
        confidence=0.0,
        reliability=1.0,
    )

    assert (
        calculate_evidence_strength(evidence)
        == pytest.approx(0.0)
    )


def test_evidence_strength_with_zero_reliability():
    evidence = create_evidence(
        confidence=1.0,
        reliability=0.0,
    )

    assert (
        calculate_evidence_strength(evidence)
        == pytest.approx(0.0)
    )


def test_aggregate_evidence_strength():
    evidence = [
        create_evidence(
            confidence=0.8,
            reliability=0.9,
        ),
        create_evidence(
            confidence=0.6,
            reliability=0.5,
        ),
    ]

    expected = (
        (0.8 * 0.9)
        + (0.6 * 0.5)
    ) / 2

    assert (
        calculate_aggregate_evidence_strength(evidence)
        == pytest.approx(expected)
    )


def test_empty_evidence_strength():
    assert (
        calculate_aggregate_evidence_strength([])
        == pytest.approx(0.0)
    )


def test_confidence_assessment():
    assessment = ConfidenceAssessment(
        confidence=0.9,
        uncertainty=0.1,
        evidence_strength=0.72,
        evidence_count=2,
    )

    assert assessment.confidence == pytest.approx(0.9)
    assert assessment.uncertainty == pytest.approx(0.1)
    assert assessment.evidence_strength == pytest.approx(0.72)
    assert assessment.evidence_count == 2


def test_confidence_assessment_invalid_confidence():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        ConfidenceAssessment(
            confidence=1.5,
            uncertainty=0.1,
            evidence_strength=0.5,
            evidence_count=1,
        )


def test_confidence_assessment_invalid_count():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        ConfidenceAssessment(
            confidence=0.9,
            uncertainty=0.1,
            evidence_strength=0.5,
            evidence_count=-1,
        )


def test_assess_agent_result():
    evidence = [
        create_evidence(
            confidence=0.8,
            reliability=0.9,
        )
    ]

    result = create_result(
        probability=0.9,
        evidence=evidence,
    )

    assessment = assess_agent_result(result)

    assert assessment.confidence == pytest.approx(0.9)
    assert assessment.uncertainty == pytest.approx(0.1)
    assert assessment.evidence_strength == pytest.approx(0.72)
    assert assessment.evidence_count == 1


def test_assess_legitimate_result():
    result = create_result(
        probability=0.1,
    )

    assessment = assess_agent_result(result)

    assert assessment.confidence == pytest.approx(0.9)
    assert assessment.uncertainty == pytest.approx(0.1)


def test_assess_uncertain_result():
    result = create_result(
        probability=0.5,
    )

    assessment = assess_agent_result(result)

    assert assessment.confidence == pytest.approx(0.5)
    assert assessment.uncertainty == pytest.approx(0.5)