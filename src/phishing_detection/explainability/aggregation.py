"""
Multi-agent explanation aggregation.

Phase 4.7
---------
Combines explanations produced by multiple specialized agents while
preserving:

- agent identity
- explainer identity
- explanation direction
- feature importance
- provenance
- individual explanation metadata

The aggregator is intentionally model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


@dataclass
class AggregatedFactor:
    """
    Internal representation of an aggregated explanation factor.
    """

    feature: str
    importance: float
    direction: ExplanationDirection

    sources: list[str] = field(
        default_factory=list
    )

    explainers: list[str] = field(
        default_factory=list
    )

    agents: list[str] = field(
        default_factory=list
    )

    contributions: list[float] = field(
        default_factory=list
    )

    original_values: list[Any] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class MultiAgentExplanationAggregator:
    """
    Aggregate explanations from multiple specialized agents.

    The aggregator accepts UnifiedExplanation objects from:

    - Email SHAP
    - Email LIME
    - URL SHAP
    - URL LIME
    - Sender native rules

    It produces one UnifiedExplanation containing the strongest
    combined evidence.
    """

    def __init__(
        self,
        max_display: int = 20,
        merge_similar_features: bool = True,
    ) -> None:

        if max_display <= 0:
            raise ValueError(
                "max_display must be greater than 0."
            )

        self.max_display = int(
            max_display
        )

        self.merge_similar_features = bool(
            merge_similar_features
        )

        self.explainer_name = (
            "multi_agent_explanation_aggregator"
        )

        self.method = "multi_agent_aggregation"

        self.method_name = self.method

        self.framework = "custom"

        self.framework_name = "custom"

    # ------------------------------------------------------------------
    # Public metadata
    # ------------------------------------------------------------------

    def get_explainer_name(self) -> str:
        return self.explainer_name

    def get_method_name(self) -> str:
        return self.method

    def get_framework_name(self) -> str:
        return self.framework_name

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def can_explain(
        self,
        explanations: Iterable[
            UnifiedExplanation
        ],
    ) -> bool:

        if explanations is None:
            return False

        explanations = list(
            explanations
        )

        if not explanations:
            return False

        return all(
            isinstance(
                explanation,
                UnifiedExplanation,
            )
            for explanation in explanations
        )

    def validate_inputs(
        self,
        explanations: Iterable[
            UnifiedExplanation
        ],
    ) -> list[UnifiedExplanation]:

        if explanations is None:
            raise ValueError(
                "explanations cannot be None."
            )

        explanations = list(
            explanations
        )

        if not explanations:
            raise ValueError(
                "At least one explanation is required."
            )

        if not self.can_explain(
            explanations
        ):
            raise ValueError(
                "All items must be UnifiedExplanation objects."
            )

        return explanations

    # ------------------------------------------------------------------
    # Feature normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_feature(
        feature: Any,
    ) -> str:

        text = str(feature).strip().lower()

        return " ".join(
            text.split()
        )

    @staticmethod
    def _agent_name(
        explanation: UnifiedExplanation,
    ) -> str:

        metadata = (
            explanation.metadata
            or {}
        )

        agent_name = metadata.get(
            "agent_name"
        )

        if agent_name:
            return str(
                agent_name
            )

        return (
            explanation.model_name
            or "unknown_agent"
        )

    @staticmethod
    def _importance(
        factor: ExplanationFactor,
    ) -> float:

        try:
            return abs(
                float(
                    factor.importance
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _collect_factors(
        self,
        explanations: list[
            UnifiedExplanation
        ],
    ) -> dict[
        tuple[str, ExplanationDirection],
        AggregatedFactor,
    ]:

        aggregated: dict[
            tuple[str, ExplanationDirection],
            AggregatedFactor,
        ] = {}

        for explanation in explanations:

            agent_name = (
                self._agent_name(
                    explanation
                )
            )

            for factor in (
                explanation.factors
                or []
            ):

                feature = (
                    self._normalize_feature(
                        factor.feature
                    )
                )

                direction = (
                    factor.direction
                )

                key = (
                    feature,
                    direction,
                )

                if key not in aggregated:

                    aggregated[key] = (
                        AggregatedFactor(
                            feature=feature,
                            importance=0.0,
                            direction=direction,
                        )
                    )

                item = aggregated[key]

                importance = (
                    self._importance(
                        factor
                    )
                )

                item.importance += (
                    importance
                )

                if (
                    agent_name
                    not in item.agents
                ):
                    item.agents.append(
                        agent_name
                    )

                if (
                    explanation.explainer_name
                    not in item.explainers
                ):
                    item.explainers.append(
                        explanation.explainer_name
                    )

                source = (
                    factor.metadata.get(
                        "source"
                    )
                    if factor.metadata
                    else None
                )

                if source is None:
                    source = (
                        explanation.explainer_name
                    )

                if (
                    source
                    not in item.sources
                ):
                    item.sources.append(
                        str(source)
                    )

                contribution = (
                    factor.contribution
                )

                if contribution is not None:

                    try:
                        item.contributions.append(
                            float(
                                contribution
                            )
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                item.original_values.append(
                    factor.value
                )

        return aggregated

    def _build_factors(
        self,
        aggregated: dict[
            tuple[str, ExplanationDirection],
            AggregatedFactor,
        ],
    ) -> list[
        ExplanationFactor
    ]:

        ordered = sorted(
            aggregated.values(),
            key=lambda item: (
                item.importance,
                len(item.agents),
            ),
            reverse=True,
        )

        ordered = ordered[
            : self.max_display
        ]

        factors: list[
            ExplanationFactor
        ] = []

        for rank, item in enumerate(
            ordered,
            start=1,
        ):

            factor_metadata = {
                "agents": list(
                    item.agents
                ),
                "explainers": list(
                    item.explainers
                ),
                "sources": list(
                    item.sources
                ),
                "support_count": len(
                    item.agents
                ),
                "provenance_count": len(
                    item.explainers
                ),
                "aggregation": (
                    "summed_absolute_importance"
                ),
            }

            factors.append(
                ExplanationFactor(
                    feature=item.feature,
                    value=(
                        item.original_values[0]
                        if item.original_values
                        else None
                    ),
                    importance=item.importance,
                    direction=item.direction,
                    rank=rank,
                    contribution=(
                        sum(
                            item.contributions
                        )
                        if item.contributions
                        else item.importance
                    ),
                    metadata=factor_metadata,
                )
            )

        return factors

    # ------------------------------------------------------------------
    # Target aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_target(
        explanations: list[
            UnifiedExplanation
        ],
    ) -> ExplanationTarget:

        probabilities = [
            float(
                explanation.target.probability
            )
            for explanation in explanations
            if explanation.target
            is not None
        ]

        if not probabilities:

            return ExplanationTarget(
                label=0,
                label_name="legitimate",
                probability=0.0,
            )

        probability = (
            sum(probabilities)
            / len(probabilities)
        )

        label = int(
            probability >= 0.5
        )

        label_name = (
            "phishing"
            if label == 1
            else "legitimate"
        )

        return ExplanationTarget(
            label=label,
            label_name=label_name,
            probability=probability,
        )

    # ------------------------------------------------------------------
    # Public aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        explanations: Iterable[
            UnifiedExplanation
        ],
    ) -> UnifiedExplanation:

        explanations = (
            self.validate_inputs(
                explanations
            )
        )

        aggregated = (
            self._collect_factors(
                explanations
            )
        )

        factors = (
            self._build_factors(
                aggregated
            )
        )

        target = (
            self._aggregate_target(
                explanations
            )
        )

        phishing_factors = sum(
            factor.direction
            == ExplanationDirection.SUPPORTS_PHISHING
            for factor in factors
        )

        legitimate_factors = sum(
            factor.direction
            == ExplanationDirection.SUPPORTS_LEGITIMATE
            for factor in factors
        )

        neutral_factors = sum(
            factor.direction
            == ExplanationDirection.NEUTRAL
            for factor in factors
        )

        agent_names = sorted(
            {
                self._agent_name(
                    explanation
                )
                for explanation in explanations
            }
        )

        explainer_names = sorted(
            {
                explanation.explainer_name
                for explanation in explanations
            }
        )

        summary = (
            "Aggregated explanations from "
            f"{len(agent_names)} agent(s) and "
            f"{len(explainer_names)} explainer(s). "
            f"The combined evidence contains "
            f"{phishing_factors} factor(s) supporting phishing, "
            f"{legitimate_factors} factor(s) supporting legitimate "
            f"classification, and "
            f"{neutral_factors} neutral factor(s)."
        )

        metadata = {
            "phase": "4.7",
            "aggregation_type": (
                "multi_agent_evidence_aggregation"
            ),
            "agent_count": len(
                agent_names
            ),
            "explainer_count": len(
                explainer_names
            ),
            "agent_names": agent_names,
            "explainer_names": explainer_names,
            "input_explanation_count": len(
                explanations
            ),
            "output_factor_count": len(
                factors
            ),
            "max_display": self.max_display,
        }

        return UnifiedExplanation(
            summary=summary,
            explainer_name=self.explainer_name,
            method=self.method,
            framework=self.framework,
            scope=ExplanationScope.LOCAL,
            target=target,
            factors=factors,
            model_name="multi_agent",
            execution_time_ms=0.0,
            metadata=metadata,
        )

    def explain(
        self,
        explanations: Iterable[
            UnifiedExplanation
        ],
    ) -> UnifiedExplanation:

        return self.aggregate(
            explanations
        )


__all__ = [
    "AggregatedFactor",
    "MultiAgentExplanationAggregator",
]