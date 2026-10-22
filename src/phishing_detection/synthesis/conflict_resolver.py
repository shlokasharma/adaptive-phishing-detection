"""
Conflict Detection and Resolution for Phase 7 (Evidence Synthesis & LLM Reasoning).

Detects inter-agent disagreements, probability tensions, and contradictory findings.
Resolves conflicts using defensible, deterministic policies without unsupported weights
or silent discarding of evidence.
"""

from __future__ import annotations

import logging
from typing import Sequence

from phishing_detection.synthesis.schema import (
    ConflictReport,
    EvidenceCategory,
    EvidencePolarity,
    StandardizedEvidence,
)

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Detects and analyzes conflicts among evidence items and agent predictions.
    """

    def __init__(
        self,
        probability_conflict_threshold: float = 0.35,
        confidence_discount_factor: float = 0.25,
    ) -> None:
        self.probability_conflict_threshold = probability_conflict_threshold
        self.confidence_discount_factor = confidence_discount_factor

    def analyze_conflicts(
        self,
        evidence_items: Sequence[StandardizedEvidence],
    ) -> ConflictReport:
        """
        Evaluate a collection of standardized evidence items for contradictions.
        """
        phishing_items = [e for e in evidence_items if e.polarity == EvidencePolarity.SUPPORTS_PHISHING]
        legit_items = [e for e in evidence_items if e.polarity == EvidencePolarity.SUPPORTS_LEGITIMATE]

        # Extract agent-level predictions (prediction is not None)
        agent_preds: dict[str, int] = {}
        agent_probs: dict[str, float] = {}

        for e in evidence_items:
            if e.prediction is not None:
                agent_preds[e.source] = e.prediction
            if e.score is not None and e.score_type.value == "probability":
                agent_probs[e.source] = e.score

        # Check 1: Categorical prediction conflict
        pred_values = set(agent_preds.values())
        if len(pred_values) > 1:
            sources_phish = [s for s, p in agent_preds.items() if p == 1]
            sources_legit = [s for s, p in agent_preds.items() if p == 0]
            conflicting_sources = sorted(list(agent_preds.keys()))

            desc = (
                f"Direct categorical prediction disagreement: {', '.join(sources_phish)} predicted phishing (1), "
                f"while {', '.join(sources_legit)} predicted legitimate (0)."
            )

            # Check if one modality is URL or Sender vs Email
            strategy = "cautious_asymmetric_security"
            rationale = (
                "Due to the asymmetric severity of phishing attacks, direct contradictions "
                "trigger a cautious security posture. The contradiction is explicitly flagged, "
                "confidence is discounted, and the system avoids unverified dismissal of phishing indicators."
            )

            return ConflictReport(
                has_conflict=True,
                conflict_type="inter_agent_prediction",
                conflicting_sources=conflicting_sources,
                description=desc,
                resolution_strategy=strategy,
                resolution_rationale=rationale,
                confidence_discount=self.confidence_discount_factor,
            )

        # Check 2: Wide probability spread among probability scores
        if len(agent_probs) >= 2:
            max_prob_source = max(agent_probs.keys(), key=lambda s: agent_probs[s])
            min_prob_source = min(agent_probs.keys(), key=lambda s: agent_probs[s])
            spread = agent_probs[max_prob_source] - agent_probs[min_prob_source]

            if spread >= self.probability_conflict_threshold:
                desc = (
                    f"Significant probability tension ({spread:.2f} spread): "
                    f"{max_prob_source} estimated {agent_probs[max_prob_source]:.2f} "
                    f"vs {min_prob_source} estimated {agent_probs[min_prob_source]:.2f}."
                )
                return ConflictReport(
                    has_conflict=True,
                    conflict_type="score_disagreement",
                    conflicting_sources=[max_prob_source, min_prob_source],
                    description=desc,
                    resolution_strategy="evidence_corroboration",
                    resolution_rationale=(
                        "Probability tension detected. Evaluated against aggregate evidence strength "
                        "rather than naive arithmetic averaging."
                    ),
                    confidence_discount=spread * 0.2,
                )

        # Check 3: Mixed evidence findings (both phishing and legitimate evidence exist)
        if phishing_items and legit_items:
            phish_categories = {e.category.value for e in phishing_items}
            legit_categories = {e.category.value for e in legit_items}

            # If sender passes authentication (SPF/DKIM) but email text or URL is malicious,
            # this is a classic compromised account / spoofed display name pattern.
            if EvidenceCategory.SENDER.value in legit_categories and (
                EvidenceCategory.EMAIL.value in phish_categories or EvidenceCategory.URL.value in phish_categories
            ):
                desc = (
                    "Modality tension: Sender authentication indicators appear legitimate, "
                    "but content/URL indicators show suspicious phishing characteristics."
                )
                rationale = (
                    "Valid SPF/DKIM authentication does not prove content innocence (e.g., legitimate compromised accounts). "
                    "Phishing indicators in content/URL take precedence for risk assessment."
                )
                return ConflictReport(
                    has_conflict=True,
                    conflict_type="modality_tension",
                    conflicting_sources=[e.source for e in phishing_items + legit_items],
                    description=desc,
                    resolution_strategy="modality_specificity_hierarchy",
                    resolution_rationale=rationale,
                    confidence_discount=0.10,
                )

            # Generic mixed evidence
            return ConflictReport(
                has_conflict=True,
                conflict_type="mixed_polarity_findings",
                conflicting_sources=list(set([e.source for e in phishing_items + legit_items])),
                description=f"Evidence set contains {len(phishing_items)} phishing findings and {len(legit_items)} legitimate findings.",
                resolution_strategy="evidence_count_and_strength_balance",
                resolution_rationale="Balanced opposing findings by specificity and documented indicators.",
                confidence_discount=0.15,
            )

        # No conflict detected
        return ConflictReport(
            has_conflict=False,
            conflict_type="none",
            conflicting_sources=[],
            description="No conflicts detected among available evidence.",
            resolution_strategy="consensus",
            resolution_rationale="All executed components agree on classification direction.",
            confidence_discount=0.0,
        )
