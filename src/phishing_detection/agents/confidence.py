"""
Confidence and uncertainty utilities for Phase 3 agents.

This module provides common confidence calculations used by
specialized phishing detection agents and the orchestration layer.

Definitions
-----------
Confidence:
    Confidence associated with the predicted class.

Uncertainty:
    Complement of confidence.

Evidence strength:
    Confidence multiplied by evidence reliability.

The utilities are intentionally model-independent so that
future reinforcement-learning policies can consume the same
signals regardless of which agent generated them.
"""

from __future__ import annotations

from dataclasses import dataclass

from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult


@dataclass(frozen=True)
class ConfidenceAssessment:
    """
    Standardized confidence assessment.

    Parameters
    ----------
    confidence:
        Confidence in the predicted class.

    uncertainty:
        Uncertainty associated with the prediction.

    evidence_strength:
        Aggregate strength of the available evidence.

    evidence_count:
        Number of evidence items considered.
    """

    confidence: float
    uncertainty: float
    evidence_strength: float
    evidence_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate assessment values."""

        for field_name, value in (
            ("confidence", self.confidence),
            ("uncertainty", self.uncertainty),
            ("evidence_strength", self.evidence_strength),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field_name} must be a number."
                )

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if not isinstance(self.evidence_count, int):
            raise TypeError(
                "evidence_count must be an integer."
            )

        if self.evidence_count < 0:
            raise ValueError(
                "evidence_count cannot be negative."
            )


def calculate_confidence(
    phishing_probability: float,
) -> float:
    """
    Calculate confidence from phishing probability.

    Confidence is defined as the probability of the predicted
    class.

    Examples
    --------
    p(phishing) = 0.90
        confidence = 0.90

    p(phishing) = 0.10
        confidence = 0.90

    p(phishing) = 0.50
        confidence = 0.50
    """

    probability = float(phishing_probability)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "phishing_probability must be between 0 and 1."
        )

    return round(
        max(probability, 1.0 - probability),
        10,
    )


def calculate_uncertainty(
    confidence: float,
) -> float:
    """
    Calculate uncertainty as the complement of confidence.
    """

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    return round(
        1.0 - confidence,
        10,
    )


def calculate_evidence_strength(
    evidence: Evidence,
) -> float:
    """
    Calculate the strength of one evidence item.

    Evidence strength is defined as:

        confidence × reliability

    Missing confidence or reliability is treated as zero.
    """

    confidence = float(
        evidence.confidence or 0.0
    )

    reliability = float(
        evidence.reliability or 0.0
    )

    return round(
        confidence * reliability,
        10,
    )


def calculate_aggregate_evidence_strength(
    evidence: list[Evidence],
) -> float:
    """
    Calculate aggregate evidence strength.

    The mean evidence strength is used so that adding more
    evidence items does not automatically inflate the score.
    """

    if not evidence:
        return 0.0

    strengths = [
        calculate_evidence_strength(item)
        for item in evidence
    ]

    return round(
        sum(strengths) / len(strengths),
        10,
    )


def assess_agent_result(
    result: AgentResult,
) -> ConfidenceAssessment:
    """
    Produce a standardized confidence assessment for an AgentResult.
    """

    confidence = calculate_confidence(
        result.phishing_probability
    )

    uncertainty = calculate_uncertainty(
        confidence
    )

    evidence_strength = (
        calculate_aggregate_evidence_strength(
            result.evidence
        )
    )

    return ConfidenceAssessment(
        confidence=confidence,
        uncertainty=uncertainty,
        evidence_strength=evidence_strength,
        evidence_count=len(result.evidence),
    )