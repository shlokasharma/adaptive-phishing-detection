"""
Phase 4.9 - Explainable AI Evaluation.

This module provides quantitative evaluation utilities for the Phase 4 XAI
layer.

Evaluated properties
--------------------
1. Explanation consistency
2. SHAP/LIME agreement
3. Explanation sparsity
4. Runtime / computational cost
5. Fidelity through perturbation-based evaluation
6. Robustness under repeated explanations
7. Multi-agent explanation coverage

The evaluator is intentionally model-agnostic. It operates on the unified
Phase 4 explanation schema and accepts callbacks for model prediction and
input perturbation where fidelity evaluation is possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from phishing_detection.explainability.schema import (
    ExplanationDirection,
    UnifiedExplanation,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FidelityResult:
    """
    Result of perturbation-based explanation fidelity evaluation.
    """

    original_probability: float
    perturbed_probability: float
    probability_change: float
    explanation_importance: float
    fidelity_score: float
    top_k: int
    direction_preserved: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_probability": self.original_probability,
            "perturbed_probability": self.perturbed_probability,
            "probability_change": self.probability_change,
            "explanation_importance": self.explanation_importance,
            "fidelity_score": self.fidelity_score,
            "top_k": self.top_k,
            "direction_preserved": self.direction_preserved,
            "metadata": self.metadata,
        }


@dataclass
class ConsistencyResult:
    """
    Result of repeated explanation consistency evaluation.
    """

    runs: int
    factor_overlap: float
    directional_agreement: float
    importance_correlation: float
    mean_factor_count: float
    factor_count_std: float
    consistency_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "factor_overlap": self.factor_overlap,
            "directional_agreement": self.directional_agreement,
            "importance_correlation": self.importance_correlation,
            "mean_factor_count": self.mean_factor_count,
            "factor_count_std": self.factor_count_std,
            "consistency_score": self.consistency_score,
            "metadata": self.metadata,
        }


@dataclass
class AgreementResult:
    """
    SHAP/LIME explanation agreement result.
    """

    top_k: int
    factor_overlap: float
    directional_agreement: float
    importance_correlation: float
    agreement_score: float
    shap_factor_count: int
    lime_factor_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "factor_overlap": self.factor_overlap,
            "directional_agreement": self.directional_agreement,
            "importance_correlation": self.importance_correlation,
            "agreement_score": self.agreement_score,
            "shap_factor_count": self.shap_factor_count,
            "lime_factor_count": self.lime_factor_count,
            "metadata": self.metadata,
        }


@dataclass
class RuntimeResult:
    """
    Runtime evaluation result.
    """

    runs: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "std_ms": self.std_ms,
            "metadata": self.metadata,
        }


@dataclass
class SparsityResult:
    """
    Explanation sparsity result.
    """

    factor_count: int
    non_zero_factor_count: int
    sparsity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_count": self.factor_count,
            "non_zero_factor_count": self.non_zero_factor_count,
            "sparsity": self.sparsity,
            "metadata": self.metadata,
        }


@dataclass
class RobustnessResult:
    """
    Robustness result across perturbed/repeated explanations.
    """

    runs: int
    successful_runs: int
    failure_count: int
    factor_overlap: float
    probability_std: float
    robustness_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "successful_runs": self.successful_runs,
            "failure_count": self.failure_count,
            "factor_overlap": self.factor_overlap,
            "probability_std": self.probability_std,
            "robustness_score": self.robustness_score,
            "metadata": self.metadata,
        }


@dataclass
class XAIQualityReport:
    """
    Complete Phase 4.9 XAI evaluation report.
    """

    fidelity: list[FidelityResult] = field(default_factory=list)
    consistency: list[ConsistencyResult] = field(
        default_factory=list
    )
    agreement: list[AgreementResult] = field(
        default_factory=list
    )
    runtime: list[RuntimeResult] = field(
        default_factory=list
    )
    sparsity: list[SparsityResult] = field(
        default_factory=list
    )
    robustness: list[RobustnessResult] = field(
        default_factory=list
    )
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity": [
                item.to_dict()
                for item in self.fidelity
            ],
            "consistency": [
                item.to_dict()
                for item in self.consistency
            ],
            "agreement": [
                item.to_dict()
                for item in self.agreement
            ],
            "runtime": [
                item.to_dict()
                for item in self.runtime
            ],
            "sparsity": [
                item.to_dict()
                for item in self.sparsity
            ],
            "robustness": [
                item.to_dict()
                for item in self.robustness
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        return 0.0

    return float(mean(values))


def _safe_std(values: Iterable[float]) -> float:
    values = list(values)

    if len(values) <= 1:
        return 0.0

    return float(pstdev(values))


def _factor_key(factor: Any) -> str:
    """
    Build a stable factor identity.
    """

    feature = str(
        getattr(factor, "feature", "")
    ).strip().lower()

    direction = str(
        getattr(factor, "direction", "")
    ).strip().lower()

    return f"{feature}|{direction}"


def _factor_feature_key(factor: Any) -> str:
    return str(
        getattr(factor, "feature", "")
    ).strip().lower()


def _factor_importance(factor: Any) -> float:
    try:
        return abs(
            float(
                getattr(
                    factor,
                    "importance",
                    0.0,
                )
            )
        )
    except (TypeError, ValueError):
        return 0.0


def _top_factors(
    explanation: UnifiedExplanation,
    top_k: int,
) -> list[Any]:
    factors = list(
        getattr(
            explanation,
            "factors",
            [],
        )
        or []
    )

    factors.sort(
        key=_factor_importance,
        reverse=True,
    )

    return factors[: max(1, top_k)]


def _pearson(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    if len(x) != len(y):
        return 0.0

    if len(x) < 2:
        return 0.0

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )

    denominator_x = sqrt(
        sum(
            (a - mean_x) ** 2
            for a in x
        )
    )

    denominator_y = sqrt(
        sum(
            (b - mean_y) ** 2
            for b in y
        )
    )

    denominator = denominator_x * denominator_y

    if denominator == 0:
        return 1.0 if x == y else 0.0

    return _clamp(
        (numerator / denominator + 1.0) / 2.0
    )


def _directional_agreement(
    first: UnifiedExplanation,
    second: UnifiedExplanation,
) -> float:
    first_map = {
        _factor_feature_key(factor): str(
            getattr(
                factor,
                "direction",
                "",
            )
        )
        for factor in getattr(first, "factors", [])
    }

    second_map = {
        _factor_feature_key(factor): str(
            getattr(
                factor,
                "direction",
                "",
            )
        )
        for factor in getattr(second, "factors", [])
    }

    common = set(first_map) & set(second_map)

    if not common:
        return 0.0

    agreements = sum(
        first_map[key] == second_map[key]
        for key in common
    )

    return agreements / len(common)


def _factor_overlap(
    first: UnifiedExplanation,
    second: UnifiedExplanation,
    top_k: int,
) -> float:
    first_keys = {
        _factor_feature_key(factor)
        for factor in _top_factors(first, top_k)
    }

    second_keys = {
        _factor_feature_key(factor)
        for factor in _top_factors(second, top_k)
    }

    if not first_keys and not second_keys:
        return 1.0

    union = first_keys | second_keys

    if not union:
        return 0.0

    return len(first_keys & second_keys) / len(union)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class XAIQualityEvaluator:
    """
    Quantitative evaluator for Phase 4 explanations.
    """

    def __init__(
        self,
        top_k: int = 5,
    ) -> None:

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        self.top_k = top_k

    # ------------------------------------------------------------------
    # Sparsity
    # ------------------------------------------------------------------

    def evaluate_sparsity(
        self,
        explanation: UnifiedExplanation,
    ) -> SparsityResult:
        factors = list(
            getattr(
                explanation,
                "factors",
                [],
            )
            or []
        )

        non_zero = sum(
            _factor_importance(factor) > 0
            for factor in factors
        )

        total = len(factors)

        sparsity = (
            1.0 - (non_zero / total)
            if total
            else 1.0
        )

        return SparsityResult(
            factor_count=total,
            non_zero_factor_count=non_zero,
            sparsity=_clamp(sparsity),
            metadata={
                "explainer_name": getattr(
                    explanation,
                    "explainer_name",
                    None,
                ),
                "method": getattr(
                    explanation,
                    "method",
                    None,
                ),
            },
        )

    # ------------------------------------------------------------------
    # SHAP/LIME agreement
    # ------------------------------------------------------------------

    def compare_explainers(
        self,
        shap_explanation: UnifiedExplanation,
        lime_explanation: UnifiedExplanation,
    ) -> AgreementResult:
        shap_factors = _top_factors(
            shap_explanation,
            self.top_k,
        )

        lime_factors = _top_factors(
            lime_explanation,
            self.top_k,
        )

        shap_map = {
            _factor_feature_key(factor): _factor_importance(factor)
            for factor in shap_factors
        }

        lime_map = {
            _factor_feature_key(factor): _factor_importance(factor)
            for factor in lime_factors
        }

        common = sorted(
            set(shap_map) & set(lime_map)
        )

        if common:
            shap_values = [
                shap_map[key]
                for key in common
            ]

            lime_values = [
                lime_map[key]
                for key in common
            ]

            importance_correlation = _pearson(
                shap_values,
                lime_values,
            )
        else:
            importance_correlation = 0.0

        overlap = _factor_overlap(
            shap_explanation,
            lime_explanation,
            self.top_k,
        )

        direction_agreement = _directional_agreement(
            shap_explanation,
            lime_explanation,
        )

        agreement_score = _safe_mean(
            [
                overlap,
                direction_agreement,
                importance_correlation,
            ]
        )

        return AgreementResult(
            top_k=self.top_k,
            factor_overlap=overlap,
            directional_agreement=direction_agreement,
            importance_correlation=importance_correlation,
            agreement_score=agreement_score,
            shap_factor_count=len(
                getattr(
                    shap_explanation,
                    "factors",
                    [],
                )
            ),
            lime_factor_count=len(
                getattr(
                    lime_explanation,
                    "factors",
                    [],
                )
            ),
            metadata={
                "comparison": "SHAP_vs_LIME",
            },
        )

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def evaluate_consistency(
        self,
        explanations: Sequence[UnifiedExplanation],
    ) -> ConsistencyResult:
        explanations = list(explanations)

        if not explanations:
            raise ValueError(
                "At least one explanation is required."
            )

        if len(explanations) == 1:
            return ConsistencyResult(
                runs=1,
                factor_overlap=1.0,
                directional_agreement=1.0,
                importance_correlation=1.0,
                mean_factor_count=float(
                    len(explanations[0].factors)
                ),
                factor_count_std=0.0,
                consistency_score=1.0,
            )

        overlaps = []
        directional_scores = []
        correlations = []

        reference = explanations[0]

        reference_map = {
            _factor_feature_key(factor): _factor_importance(factor)
            for factor in getattr(
                reference,
                "factors",
                [],
            )
        }

        for explanation in explanations[1:]:
            overlaps.append(
                _factor_overlap(
                    reference,
                    explanation,
                    self.top_k,
                )
            )

            directional_scores.append(
                _directional_agreement(
                    reference,
                    explanation,
                )
            )

            current_map = {
                _factor_feature_key(factor): _factor_importance(factor)
                for factor in getattr(
                    explanation,
                    "factors",
                    [],
                )
            }

            common = sorted(
                set(reference_map)
                & set(current_map)
            )

            if common:
                correlations.append(
                    _pearson(
                        [
                            reference_map[key]
                            for key in common
                        ],
                        [
                            current_map[key]
                            for key in common
                        ],
                    )
                )

        factor_counts = [
            len(
                getattr(
                    explanation,
                    "factors",
                    [],
                )
            )
            for explanation in explanations
        ]

        overlap = _safe_mean(overlaps)
        direction = _safe_mean(
            directional_scores
        )
        correlation = _safe_mean(
            correlations
        )

        consistency_score = _safe_mean(
            [
                overlap,
                direction,
                correlation,
            ]
        )

        return ConsistencyResult(
            runs=len(explanations),
            factor_overlap=overlap,
            directional_agreement=direction,
            importance_correlation=correlation,
            mean_factor_count=_safe_mean(
                factor_counts
            ),
            factor_count_std=_safe_std(
                factor_counts
            ),
            consistency_score=consistency_score,
        )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def evaluate_runtime(
        self,
        explanation_fn: Callable[[], Any],
        runs: int = 3,
    ) -> RuntimeResult:
        if runs < 1:
            raise ValueError(
                "runs must be >= 1."
            )

        durations_ms = []

        for _ in range(runs):
            start = perf_counter()

            explanation_fn()

            elapsed = (
                perf_counter() - start
            ) * 1000.0

            durations_ms.append(elapsed)

        ordered = sorted(durations_ms)

        if len(ordered) % 2:
            median = ordered[
                len(ordered) // 2
            ]
        else:
            middle = len(ordered) // 2
            median = (
                ordered[middle - 1]
                + ordered[middle]
            ) / 2.0

        return RuntimeResult(
            runs=runs,
            mean_ms=_safe_mean(
                durations_ms
            ),
            median_ms=float(median),
            min_ms=min(durations_ms),
            max_ms=max(durations_ms),
            std_ms=_safe_std(
                durations_ms
            ),
        )

    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------

    def evaluate_fidelity(
        self,
        explanation: UnifiedExplanation,
        original_probability: float,
        perturbed_probability: float,
        top_k: int | None = None,
    ) -> FidelityResult:
        """
        Evaluate perturbation-based fidelity.

        A fidelity score measures how strongly the prediction changes when
        the features identified as important by the explanation are removed
        or perturbed.

        The actual perturbation is performed outside this evaluator so that
        email-token and URL-feature modalities can each define an appropriate
        perturbation strategy.
        """

        k = (
            self.top_k
            if top_k is None
            else max(1, top_k)
        )

        factors = _top_factors(
            explanation,
            k,
        )

        importance = _safe_mean(
            [
                _factor_importance(factor)
                for factor in factors
            ]
        )

        probability_change = abs(
            float(original_probability)
            - float(perturbed_probability)
        )

        fidelity_score = _clamp(
            probability_change
            * (1.0 + importance)
        )

        direction = getattr(
            explanation,
            "target",
            None,
        )

        direction_preserved = (
            probability_change > 0
            if direction is not None
            else False
        )

        return FidelityResult(
            original_probability=float(
                original_probability
            ),
            perturbed_probability=float(
                perturbed_probability
            ),
            probability_change=probability_change,
            explanation_importance=importance,
            fidelity_score=fidelity_score,
            top_k=k,
            direction_preserved=direction_preserved,
        )

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def evaluate_robustness(
        self,
        explanation_fn: Callable[[], UnifiedExplanation],
        prediction_fn: Callable[[], float] | None = None,
        runs: int = 5,
    ) -> RobustnessResult:
        if runs < 1:
            raise ValueError(
                "runs must be >= 1."
            )

        explanations: list[
            UnifiedExplanation
        ] = []

        probabilities: list[float] = []

        failures = 0

        for _ in range(runs):
            try:
                explanation = explanation_fn()

                if not isinstance(
                    explanation,
                    UnifiedExplanation,
                ):
                    raise TypeError(
                        "explanation_fn must return "
                        "UnifiedExplanation."
                    )

                explanations.append(
                    explanation
                )

                if prediction_fn is not None:
                    probabilities.append(
                        float(
                            prediction_fn()
                        )
                    )

            except Exception:
                failures += 1

        successful = len(explanations)

        if successful <= 1:
            overlap = (
                1.0
                if successful == 1
                else 0.0
            )
        else:
            pairwise = []

            for index in range(
                1,
                successful,
            ):
                pairwise.append(
                    _factor_overlap(
                        explanations[0],
                        explanations[index],
                        self.top_k,
                    )
                )

            overlap = _safe_mean(
                pairwise
            )

        probability_std = _safe_std(
            probabilities
        )

        stability = _clamp(
            1.0
            - (
                probability_std
                / max(
                    1.0,
                    max(
                        probabilities,
                        default=1.0,
                    ),
                )
            )
        )

        success_rate = (
            successful / runs
        )

        robustness_score = _safe_mean(
            [
                overlap,
                stability,
                success_rate,
            ]
        )

        return RobustnessResult(
            runs=runs,
            successful_runs=successful,
            failure_count=failures,
            factor_overlap=overlap,
            probability_std=probability_std,
            robustness_score=robustness_score,
        )

    # ------------------------------------------------------------------
    # Complete report
    # ------------------------------------------------------------------

    def summarize(
        self,
        report: XAIQualityReport,
    ) -> dict[str, Any]:
        """
        Produce aggregate summary metrics.
        """

        summary: dict[str, Any] = {}

        if report.fidelity:
            summary["mean_fidelity"] = _safe_mean(
                item.fidelity_score
                for item in report.fidelity
            )

        if report.consistency:
            summary["mean_consistency"] = _safe_mean(
                item.consistency_score
                for item in report.consistency
            )

        if report.agreement:
            summary["mean_shap_lime_agreement"] = (
                _safe_mean(
                    item.agreement_score
                    for item in report.agreement
                )
            )

        if report.runtime:
            summary["mean_runtime_ms"] = _safe_mean(
                item.mean_ms
                for item in report.runtime
            )

        if report.sparsity:
            summary["mean_sparsity"] = _safe_mean(
                item.sparsity
                for item in report.sparsity
            )

        if report.robustness:
            summary["mean_robustness"] = _safe_mean(
                item.robustness_score
                for item in report.robustness
            )

        report.summary = summary

        return summary


__all__ = [
    "FidelityResult",
    "ConsistencyResult",
    "AgreementResult",
    "RuntimeResult",
    "SparsityResult",
    "RobustnessResult",
    "XAIQualityReport",
    "XAIQualityEvaluator",
]