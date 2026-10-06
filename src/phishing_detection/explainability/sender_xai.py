"""
Sender explainability for rule-based sender analysis.

Phase 4.5
---------
Provides transparent explanations for the SenderAnalysisAgent.

Unlike the email and URL detectors, the sender detector is
rule-based rather than a learned ML model. Therefore, the
appropriate explanation mechanism is native rule attribution.

Each sender-analysis evidence item is converted directly into
an ExplanationFactor in the unified XAI schema.
"""

from __future__ import annotations

import time
from typing import Any

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.base import ExplanationMetadata
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


class SenderXAIExplainer:
    """
    Explain SenderAnalysisAgent decisions using its rule evidence.

    The explanation is faithful to the sender agent because each
    explanation factor corresponds directly to evidence produced
    by the sender analysis rules.
    """

    def __init__(
        self,
        max_display: int = 15,
    ) -> None:

        if max_display <= 0:
            raise ValueError(
                "max_display must be greater than 0"
            )

        self.max_display = int(max_display)

        # Public metadata
        self.explainer_name = "sender_rule_explainer"
        self.method = "rule_based"
        self.method_name = self.method
        self.framework = "native"
        self.framework_name = "native"

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_explainer_name(self) -> str:
        return self.explainer_name

    def get_method_name(self) -> str:
        return self.method

    def get_framework_name(self) -> str:
        return self.framework_name

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def can_explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> bool:
        """
        Return True only for SenderAnalysisAgent results.
        """

        if result is None or context is None:
            return False

        return result.modality == "sender"

    def validate_inputs(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> None:

        if not self.can_explain(
            result,
            context,
        ):
            raise ValueError(
                "SenderXAIExplainer requires a sender "
                "AgentResult."
            )

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_description(
        evidence: Any,
    ) -> str:
        """
        Obtain a human-readable description from an Evidence object.
        """

        description = getattr(
            evidence,
            "description",
            None,
        )

        if description:
            return str(description)

        value = getattr(
            evidence,
            "value",
            None,
        )

        evidence_type = getattr(
            evidence,
            "evidence_type",
            "sender_evidence",
        )

        return (
            f"{evidence_type}: {value}"
        )

    @staticmethod
    def _direction(
        evidence: Any,
    ) -> ExplanationDirection:
        """
        Convert sender evidence polarity to XAI direction.
        """

        if getattr(
            evidence,
            "supports_phishing",
            False,
        ):
            return ExplanationDirection.SUPPORTS_PHISHING

        if getattr(
            evidence,
            "supports_legitimate",
            False,
        ):
            return ExplanationDirection.SUPPORTS_LEGITIMATE

        return ExplanationDirection.NEUTRAL

    @staticmethod
    def _importance(
        evidence: Any,
    ) -> float:
        """
        Calculate explanation importance.

        Sender evidence already contains confidence and reliability.
        Their product provides a transparent strength score.
        """

        confidence = getattr(
            evidence,
            "confidence",
            None,
        )

        reliability = getattr(
            evidence,
            "reliability",
            None,
        )

        if confidence is None:
            confidence = 1.0

        if reliability is None:
            reliability = 1.0

        confidence = max(
            0.0,
            min(1.0, float(confidence)),
        )

        reliability = max(
            0.0,
            min(1.0, float(reliability)),
        )

        return confidence * reliability

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> UnifiedExplanation:

        self.validate_inputs(
            result,
            context,
        )

        start_time = time.perf_counter()

        evidence_items = list(
            result.evidence or []
        )

        # Rank evidence by its transparent strength.
        ranked_evidence = sorted(
            evidence_items,
            key=self._importance,
            reverse=True,
        )

        ranked_evidence = ranked_evidence[
            : self.max_display
        ]

        factors: list[ExplanationFactor] = []

        for rank, evidence in enumerate(
            ranked_evidence,
            start=1,
        ):

            importance = self._importance(
                evidence
            )

            direction = self._direction(
                evidence
            )

            evidence_type = getattr(
                evidence,
                "evidence_type",
                "sender_evidence",
            )

            value = getattr(
                evidence,
                "value",
                None,
            )

            factors.append(
                ExplanationFactor(
                    feature=str(
                        evidence_type
                    ),
                    value=value,
                    importance=importance,
                    direction=direction,
                    rank=rank,
                    contribution=(
                        importance
                        if direction
                        == ExplanationDirection.SUPPORTS_PHISHING
                        else (
                            -importance
                            if direction
                            == ExplanationDirection.SUPPORTS_LEGITIMATE
                            else 0.0
                        )
                    ),
                    metadata={
                        "source": getattr(
                            evidence,
                            "source",
                            "sender_analysis_agent",
                        ),
                        "description": (
                            self._evidence_description(
                                evidence
                            )
                        ),
                        "confidence": getattr(
                            evidence,
                            "confidence",
                            None,
                        ),
                        "reliability": getattr(
                            evidence,
                            "reliability",
                            None,
                        ),
                    },
                )
            )

        phishing_count = sum(
            factor.direction
            == ExplanationDirection.SUPPORTS_PHISHING
            for factor in factors
        )

        legitimate_count = sum(
            factor.direction
            == ExplanationDirection.SUPPORTS_LEGITIMATE
            for factor in factors
        )

        neutral_count = sum(
            factor.direction
            == ExplanationDirection.NEUTRAL
            for factor in factors
        )

        if factors:
            summary = (
                "Sender rule analysis identified "
                f"{phishing_count} factor(s) supporting phishing, "
                f"{legitimate_count} factor(s) supporting "
                f"legitimate classification, and "
                f"{neutral_count} neutral factor(s)."
            )
        else:
            summary = (
                "Sender rule analysis did not produce "
                "specific evidence factors."
            )

        execution_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        metadata = ExplanationMetadata(
            explainer_name=self.explainer_name,
            method=self.method,
            model_name="sender_rule_analysis",
            framework=self.framework,
            execution_time_ms=execution_time_ms,
            additional_metadata={
                "phase": "4.5",
                "evidence_count": len(
                    evidence_items
                ),
                "displayed_factor_count": len(
                    factors
                ),
                "explanation_type": "native_rule_attribution",
            },
        )

        return UnifiedExplanation(
            summary=summary,
            explainer_name=self.explainer_name,
            method=self.method,
            framework=self.framework,
            scope=ExplanationScope.LOCAL,
            target=ExplanationTarget(
                label=result.label,
                label_name=result.label_name,
                probability=float(
                    result.phishing_probability
                ),
            ),
            factors=factors,
            model_name="sender_rule_analysis",
            execution_time_ms=execution_time_ms,
            metadata=metadata.to_dict(),
        )


__all__ = [
    "SenderXAIExplainer",
]