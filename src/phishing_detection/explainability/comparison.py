"""
SHAP vs LIME explanation comparison utilities.

Phase 4.6
---------
Provides model-agnostic quantitative comparison of SHAP and LIME
UnifiedExplanation objects.

The comparison focuses on:
    - top-factor overlap
    - factor agreement
    - importance agreement
    - directional agreement
    - explanation sparsity
    - execution time
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from phishing_detection.explainability.schema import (
    ExplanationDirection,
    UnifiedExplanation,
)


@dataclass(frozen=True)
class XAIComparisonResult:
    """
    Quantitative comparison between two explanations.

    The two explanations are normally SHAP and LIME explanations
    for the same input and model.
    """

    explainer_a: str
    explainer_b: str

    factor_count_a: int
    factor_count_b: int

    top_k: int

    top_k_overlap_count: int
    top_k_overlap_ratio: float

    directional_agreement_ratio: float
    importance_correlation: float | None

    explanation_sparsity_a: float
    explanation_sparsity_b: float

    execution_time_ms_a: float
    execution_time_ms_b: float

    metadata: dict[str, Any]

    @property
    def factor_count_difference(self) -> int:
        return abs(
            self.factor_count_a
            - self.factor_count_b
        )

    @property
    def runtime_ratio(self) -> float | None:
        """
        Runtime ratio B/A.

        Returns None if explanation A has zero runtime.
        """

        if self.execution_time_ms_a <= 0:
            return None

        return (
            self.execution_time_ms_b
            / self.execution_time_ms_a
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "explainer_a": self.explainer_a,
            "explainer_b": self.explainer_b,
            "factor_count_a": self.factor_count_a,
            "factor_count_b": self.factor_count_b,
            "top_k": self.top_k,
            "top_k_overlap_count": (
                self.top_k_overlap_count
            ),
            "top_k_overlap_ratio": (
                self.top_k_overlap_ratio
            ),
            "directional_agreement_ratio": (
                self.directional_agreement_ratio
            ),
            "importance_correlation": (
                self.importance_correlation
            ),
            "explanation_sparsity_a": (
                self.explanation_sparsity_a
            ),
            "explanation_sparsity_b": (
                self.explanation_sparsity_b
            ),
            "execution_time_ms_a": (
                self.execution_time_ms_a
            ),
            "execution_time_ms_b": (
                self.execution_time_ms_b
            ),
            "factor_count_difference": (
                self.factor_count_difference
            ),
            "runtime_ratio_b_over_a": (
                self.runtime_ratio
            ),
            "metadata": self.metadata,
        }


def _normalise_feature_name(value: Any) -> str:
    """
    Normalize feature/token names for overlap comparison.

    This intentionally performs only conservative normalization.
    It does not stem or remove meaningful words because doing so
    could artificially increase SHAP/LIME agreement.
    """

    text = str(value).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def _factor_name(factor: Any) -> str:
    return _normalise_feature_name(
        getattr(
            factor,
            "feature",
            "",
        )
    )


def _factor_importance(factor: Any) -> float:
    importance = getattr(
        factor,
        "importance",
        0.0,
    )

    try:
        return abs(float(importance))
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _factor_direction(factor: Any) -> Any:
    return getattr(
        factor,
        "direction",
        ExplanationDirection.NEUTRAL,
    )


def _top_factors(
    explanation: UnifiedExplanation,
    top_k: int,
) -> list[Any]:

    factors = list(
        explanation.factors or []
    )

    factors.sort(
        key=_factor_importance,
        reverse=True,
    )

    return factors[:top_k]


def _safe_mean(
    values: Iterable[float],
) -> float:

    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


def _pearson_correlation(
    values_a: list[float],
    values_b: list[float],
) -> float | None:
    """
    Calculate Pearson correlation without requiring scipy.

    Returns None when correlation is mathematically undefined.
    """

    if len(values_a) != len(values_b):
        raise ValueError(
            "Correlation inputs must have equal length."
        )

    if len(values_a) < 2:
        return None

    mean_a = _safe_mean(values_a)
    mean_b = _safe_mean(values_b)

    centered_a = [
        value - mean_a
        for value in values_a
    ]

    centered_b = [
        value - mean_b
        for value in values_b
    ]

    numerator = sum(
        a * b
        for a, b in zip(
            centered_a,
            centered_b,
        )
    )

    denominator_a = math.sqrt(
        sum(
            a * a
            for a in centered_a
        )
    )

    denominator_b = math.sqrt(
        sum(
            b * b
            for b in centered_b
        )
    )

    denominator = (
        denominator_a
        * denominator_b
    )

    if denominator == 0.0:
        return None

    return numerator / denominator


def _explanation_sparsity(
    explanation: UnifiedExplanation,
) -> float:
    """
    Fraction of displayed factors with non-zero importance.

    A value of 1.0 means every displayed factor contributes.
    """

    factors = list(
        explanation.factors or []
    )

    if not factors:
        return 0.0

    non_zero = sum(
        _factor_importance(factor) > 0.0
        for factor in factors
    )

    return non_zero / len(factors)


def compare_explanations(
    explanation_a: UnifiedExplanation,
    explanation_b: UnifiedExplanation,
    top_k: int = 10,
) -> XAIComparisonResult:
    """
    Compare two UnifiedExplanation objects.

    Parameters
    ----------
    explanation_a:
        Usually the SHAP explanation.

    explanation_b:
        Usually the LIME explanation.

    top_k:
        Number of highest-importance factors used for overlap
        and directional agreement.

    Returns
    -------
    XAIComparisonResult
        Structured quantitative comparison.
    """

    if explanation_a is None:
        raise ValueError(
            "explanation_a cannot be None."
        )

    if explanation_b is None:
        raise ValueError(
            "explanation_b cannot be None."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    factors_a = list(
        explanation_a.factors or []
    )

    factors_b = list(
        explanation_b.factors or []
    )

    top_a = _top_factors(
        explanation_a,
        top_k,
    )

    top_b = _top_factors(
        explanation_b,
        top_k,
    )

    names_a = {
        _factor_name(factor)
        for factor in top_a
    }

    names_b = {
        _factor_name(factor)
        for factor in top_b
    }

    overlap = names_a.intersection(
        names_b
    )

    denominator = max(
        1,
        min(
            len(names_a),
            len(names_b),
        ),
    )

    overlap_ratio = (
        len(overlap)
        / denominator
    )

    # --------------------------------------------------------------
    # Directional agreement
    # --------------------------------------------------------------

    direction_map_a = {
        _factor_name(factor): _factor_direction(
            factor
        )
        for factor in top_a
        if _factor_name(factor)
    }

    direction_map_b = {
        _factor_name(factor): _factor_direction(
            factor
        )
        for factor in top_b
        if _factor_name(factor)
    }

    common_direction_names = (
        set(direction_map_a)
        .intersection(
            direction_map_b
        )
    )

    if common_direction_names:
        directional_matches = sum(
            direction_map_a[name]
            == direction_map_b[name]
            for name in common_direction_names
        )

        directional_agreement = (
            directional_matches
            / len(common_direction_names)
        )
    else:
        directional_agreement = 0.0

    # --------------------------------------------------------------
    # Importance correlation
    # --------------------------------------------------------------

    importance_map_a = {
        _factor_name(factor): _factor_importance(
            factor
        )
        for factor in factors_a
        if _factor_name(factor)
    }

    importance_map_b = {
        _factor_name(factor): _factor_importance(
            factor
        )
        for factor in factors_b
        if _factor_name(factor)
    }

    common_importance_names = (
        set(importance_map_a)
        .intersection(
            importance_map_b
        )
    )

    if len(common_importance_names) >= 2:
        values_a = [
            importance_map_a[name]
            for name in sorted(
                common_importance_names
            )
        ]

        values_b = [
            importance_map_b[name]
            for name in sorted(
                common_importance_names
            )
        ]

        importance_correlation = (
            _pearson_correlation(
                values_a,
                values_b,
            )
        )
    else:
        importance_correlation = None

    execution_time_a = float(
        explanation_a.execution_time_ms
        or 0.0
    )

    execution_time_b = float(
        explanation_b.execution_time_ms
        or 0.0
    )

    return XAIComparisonResult(
        explainer_a=(
            explanation_a.explainer_name
        ),
        explainer_b=(
            explanation_b.explainer_name
        ),
        factor_count_a=len(factors_a),
        factor_count_b=len(factors_b),
        top_k=top_k,
        top_k_overlap_count=len(overlap),
        top_k_overlap_ratio=overlap_ratio,
        directional_agreement_ratio=(
            directional_agreement
        ),
        importance_correlation=(
            importance_correlation
        ),
        explanation_sparsity_a=(
            _explanation_sparsity(
                explanation_a
            )
        ),
        explanation_sparsity_b=(
            _explanation_sparsity(
                explanation_b
            )
        ),
        execution_time_ms_a=execution_time_a,
        execution_time_ms_b=execution_time_b,
        metadata={
            "phase": "4.6",
            "comparison_type": "shap_vs_lime",
            "common_factor_count": len(
                common_importance_names
            ),
        },
    )


__all__ = [
    "XAIComparisonResult",
    "compare_explanations",
]