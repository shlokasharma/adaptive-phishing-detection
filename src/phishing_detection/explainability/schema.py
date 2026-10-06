"""
Unified explainability schema.

This module defines standardized structures for representing
feature-level explanations produced by SHAP, LIME, and
rule-based/native explainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExplanationDirection(str, Enum):
    """
    Direction of an explanatory factor.
    """

    SUPPORTS_PHISHING = "supports_phishing"
    SUPPORTS_LEGITIMATE = "supports_legitimate"
    NEUTRAL = "neutral"


class ExplanationScope(str, Enum):
    """
    Scope of an explanation.
    """

    LOCAL = "local"
    GLOBAL = "global"


@dataclass
class ExplanationFactor:
    """
    Represents one influential explanatory factor.

    Parameters
    ----------
    feature:
        Name of the feature, token, rule, or evidence item.

    value:
        Value associated with the factor.

    importance:
        Signed or absolute importance assigned by the
        explanation method.

    direction:
        Whether the factor supports phishing, legitimacy,
        or is neutral.

    rank:
        Importance ranking among the reported factors.

    contribution:
        Optional raw contribution value returned by the
        underlying explanation framework.

    metadata:
        Additional modality-specific information.
    """

    feature: str
    value: Any
    importance: float
    direction: ExplanationDirection
    rank: int
    contribution: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature.strip():
            raise ValueError("feature must not be empty.")

        if self.rank < 1:
            raise ValueError("rank must be greater than or equal to 1.")

        if not isinstance(self.direction, ExplanationDirection):
            try:
                self.direction = ExplanationDirection(self.direction)
            except ValueError as exc:
                raise ValueError(
                    "direction must be a valid ExplanationDirection."
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Convert the factor to a serializable dictionary."""
        return {
            "feature": self.feature,
            "value": self.value,
            "importance": self.importance,
            "direction": self.direction.value,
            "rank": self.rank,
            "contribution": self.contribution,
            "metadata": self.metadata,
        }


@dataclass
class ExplanationTarget:
    """
    Describes the prediction target being explained.
    """

    label: int
    label_name: str
    probability: float

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be either 0 or 1.")

        if not self.label_name.strip():
            raise ValueError("label_name must not be empty.")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the target to a serializable dictionary."""
        return {
            "label": self.label,
            "label_name": self.label_name,
            "probability": self.probability,
        }


@dataclass
class UnifiedExplanation:
    """
    Standardized explanation representation.

    This structure is intentionally framework-neutral and can
    represent SHAP, LIME, or native/rule-based explanations.
    """

    summary: str
    explainer_name: str
    method: str
    framework: str
    scope: ExplanationScope
    target: ExplanationTarget
    factors: list[ExplanationFactor]
    model_name: str | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary must not be empty.")

        if not self.explainer_name.strip():
            raise ValueError(
                "explainer_name must not be empty."
            )

        if not self.method.strip():
            raise ValueError("method must not be empty.")

        if not self.framework.strip():
            raise ValueError("framework must not be empty.")

        if not isinstance(self.scope, ExplanationScope):
            try:
                self.scope = ExplanationScope(self.scope)
            except ValueError as exc:
                raise ValueError(
                    "scope must be a valid ExplanationScope."
                ) from exc

        if self.model_name is not None and not self.model_name.strip():
            raise ValueError(
                "model_name must not be empty when provided."
            )

        if self.execution_time_ms is not None:
            if self.execution_time_ms < 0:
                raise ValueError(
                    "execution_time_ms must be non-negative."
                )

    @property
    def factor_count(self) -> int:
        """Return the number of explanation factors."""
        return len(self.factors)

    @property
    def phishing_factors(self) -> list[ExplanationFactor]:
        """Return factors supporting the phishing class."""
        return [
            factor
            for factor in self.factors
            if factor.direction
            == ExplanationDirection.SUPPORTS_PHISHING
        ]

    @property
    def legitimate_factors(self) -> list[ExplanationFactor]:
        """Return factors supporting the legitimate class."""
        return [
            factor
            for factor in self.factors
            if factor.direction
            == ExplanationDirection.SUPPORTS_LEGITIMATE
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the unified explanation to a dictionary."""
        return {
            "summary": self.summary,
            "explainer_name": self.explainer_name,
            "method": self.method,
            "framework": self.framework,
            "scope": self.scope.value,
            "target": self.target.to_dict(),
            "factors": [
                factor.to_dict()
                for factor in self.factors
            ],
            "model_name": self.model_name,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }