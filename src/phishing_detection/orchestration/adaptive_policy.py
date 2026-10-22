"""Deterministic adaptive evidence policy for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from phishing_detection.agents.confidence import (
    calculate_aggregate_evidence_strength,
    calculate_confidence,
    calculate_uncertainty,
)


@dataclass(frozen=True)
class EvidencePolicyAssessment:
    """Assessment of the evidence collected so far."""

    phishing_probability: float
    confidence: float
    uncertainty: float
    evidence_strength: float
    evidence_count: int
    agent_count: int
    sufficient: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phishing_probability": self.phishing_probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "evidence_strength": self.evidence_strength,
            "evidence_count": self.evidence_count,
            "agent_count": self.agent_count,
            "sufficient": self.sufficient,
            "reason": self.reason,
        }


class AdaptiveEvidencePolicy:
    """
    Deterministic adaptive evidence policy for Phase 5.

    Evaluates the *combined* evidence from all agents executed so
    far and decides whether the orchestrator has sufficient
    evidence to stop or whether additional analysis is required.

    Parameters
    ----------
    confidence_threshold:
        Minimum confidence required to declare evidence
        sufficient.  Must be between 0 and 1.

    min_evidence_count:
        Minimum number of individual evidence items required
        across all executed agents.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.90,
        min_evidence_count: int = 1,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )
        if min_evidence_count < 0:
            raise ValueError(
                "min_evidence_count cannot be negative."
            )

        self.confidence_threshold = float(confidence_threshold)
        self.min_evidence_count = int(min_evidence_count)

    def assess(self, results: Sequence[Any]) -> EvidencePolicyAssessment:
        """
        Assess all evidence collected from the completed agents.

        Parameters
        ----------
        results:
            Sequence of ``AgentResult`` objects from agents that
            have already executed.

        Returns
        -------
        EvidencePolicyAssessment
            A structured assessment including the combined
            probability, confidence, uncertainty, evidence
            strength, evidence count, agent count, a boolean
            ``sufficient`` flag, and a human-readable reason.

        Notes
        -----
        Combined probability is the arithmetic mean of the
        phishing probabilities from all agent results.  This is
        the same aggregation formula used in the orchestrator's
        ``_aggregate_results`` method.
        """

        if not results:
            return EvidencePolicyAssessment(
                phishing_probability=0.5,
                confidence=0.5,
                uncertainty=0.5,
                evidence_strength=0.0,
                evidence_count=0,
                agent_count=0,
                sufficient=False,
                reason="No agent results are available; collect evidence.",
            )

        probabilities = [
            float(result.phishing_probability)
            for result in results
        ]

        for probability in probabilities:
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "Agent phishing probabilities must be between 0 and 1."
                )

        combined_probability = sum(probabilities) / len(probabilities)
        confidence = calculate_confidence(combined_probability)
        uncertainty = calculate_uncertainty(confidence)

        evidence = [
            item
            for result in results
            for item in result.evidence
        ]
        evidence_count = len(evidence)
        evidence_strength = calculate_aggregate_evidence_strength(
            evidence
        )

        # Detect contradiction among confident agents
        has_contradiction = False
        confident_results = [
            p for p in probabilities
            if calculate_confidence(p) >= self.confidence_threshold
        ]
        if len(confident_results) > 1:
            labels = {int(p >= 0.5) for p in confident_results}
            if len(labels) > 1:
                has_contradiction = True

        latest_confidence = calculate_confidence(probabilities[-1])
        latest_decisive = (
            latest_confidence >= self.confidence_threshold
            and not has_contradiction
        )

        evidence_meets_min = evidence_count >= self.min_evidence_count
        confidence_meets_threshold = (
            confidence >= self.confidence_threshold
            or latest_decisive
        )

        sufficient = bool(
            confidence_meets_threshold
            and evidence_meets_min
            and not has_contradiction
        )

        if has_contradiction:
            reason = (
                "Contradictory predictions detected among agents; "
                "additional evidence is required to resolve conflict."
            )
        elif not evidence_meets_min:
            reason = (
                "The minimum evidence requirement is not yet satisfied."
            )
        elif not confidence_meets_threshold:
            reason = (
                "Combined confidence is below the threshold; "
                "additional available evidence may help."
            )
        else:
            reason = (
                "Combined confidence meets the threshold and the "
                "minimum evidence requirement is satisfied."
            )

        return EvidencePolicyAssessment(
            phishing_probability=round(combined_probability, 10),
            confidence=confidence,
            uncertainty=uncertainty,
            evidence_strength=evidence_strength,
            evidence_count=evidence_count,
            agent_count=len(results),
            sufficient=sufficient,
            reason=reason,
        )


__all__ = [
    "AdaptiveEvidencePolicy",
    "EvidencePolicyAssessment",
]
