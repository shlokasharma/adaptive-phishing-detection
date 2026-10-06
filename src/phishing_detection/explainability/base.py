"""
Common explainability architecture for phishing detection.

This module defines the shared interface used by SHAP, LIME,
and native/rule-based explainers.

The implementation is intentionally independent of a particular
XAI framework so that future explainers can be added without
changing the downstream pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult


@dataclass
class ExplanationMetadata:
    """
    Metadata describing how an explanation was generated.
    """

    explainer_name: str
    method: str
    model_name: str | None = None
    framework: str | None = None
    execution_time_ms: float | None = None
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.explainer_name.strip():
            raise ValueError("explainer_name must not be empty.")

        if not self.method.strip():
            raise ValueError("method must not be empty.")

        if self.model_name is not None and not self.model_name.strip():
            raise ValueError(
                "model_name must not be empty when provided."
            )

        if self.framework is not None and not self.framework.strip():
            raise ValueError(
                "framework must not be empty when provided."
            )

        if self.execution_time_ms is not None:
            if self.execution_time_ms < 0:
                raise ValueError(
                    "execution_time_ms must be non-negative."
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata into a serializable dictionary."""
        return {
            "explainer_name": self.explainer_name,
            "method": self.method,
            "model_name": self.model_name,
            "framework": self.framework,
            "execution_time_ms": self.execution_time_ms,
            "additional_metadata": self.additional_metadata,
        }


@dataclass
class ExplanationResult:
    """
    Standardized explanation produced by an XAI component.

    Attributes
    ----------
    summary:
        Human-readable summary of the explanation.

    factors:
        Ordered list of influential factors. Each factor is represented
        as a dictionary so that different modalities can expose
        modality-specific information.

    prediction:
        Optional predicted phishing probability.

    metadata:
        Information about the explainer and method used.

    raw_explanation:
        Optional framework-specific representation, retained for
        research and debugging purposes.
    """

    summary: str
    factors: list[dict[str, Any]]
    metadata: ExplanationMetadata
    prediction: float | None = None
    raw_explanation: Any = None

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary must not be empty.")

        if not isinstance(self.factors, list):
            raise TypeError("factors must be a list.")

        if self.prediction is not None:
            if not 0.0 <= self.prediction <= 1.0:
                raise ValueError(
                    "prediction must be between 0 and 1."
                )

    @property
    def factor_count(self) -> int:
        """Return the number of explanation factors."""
        return len(self.factors)

    def to_dict(
        self,
        include_raw_explanation: bool = False,
    ) -> dict[str, Any]:
        """
        Convert the explanation into a serializable dictionary.

        Parameters
        ----------
        include_raw_explanation:
            Whether to include the framework-specific raw explanation.
        """
        result = {
            "summary": self.summary,
            "factors": self.factors,
            "metadata": self.metadata.to_dict(),
            "prediction": self.prediction,
        }

        if include_raw_explanation:
            result["raw_explanation"] = self.raw_explanation

        return result


class BaseExplainer(ABC):
    """
    Abstract interface for all explainability components.

    Concrete implementations may use:

    - SHAP
    - LIME
    - native model explanations
    - rule-based explanations
    - future XAI techniques

    The interface deliberately accepts both AgentResult and AgentContext
    so explanations can use the model decision as well as the original
    input and metadata.
    """

    @abstractmethod
    def get_explainer_name(self) -> str:
        """
        Return the unique explainer name.
        """
        raise NotImplementedError

    @abstractmethod
    def get_method_name(self) -> str:
        """
        Return the explanation method name.

        Examples
        --------
        shap
        lime
        rule_based
        native_feature_importance
        """
        raise NotImplementedError

    def get_framework_name(self) -> str:
        """
        Return the underlying XAI framework.

        Concrete implementations may override this method.

        Examples
        --------
        SHAP
        LIME
        Native
        """
        return "custom"

    def can_explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> bool:
        """
        Determine whether this explainer can explain the result.

        The default implementation accepts every valid AgentResult
        and AgentContext.
        """
        self.validate_inputs(result, context)
        return True

    def validate_inputs(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> None:
        """
        Validate common explainer inputs.
        """
        if not isinstance(result, AgentResult):
            raise TypeError(
                "result must be an AgentResult instance."
            )

        if not isinstance(context, AgentContext):
            raise TypeError(
                "context must be an AgentContext instance."
            )

    def create_metadata(
        self,
        *,
        model_name: str | None = None,
        execution_time_ms: float | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> ExplanationMetadata:
        """
        Create standardized explanation metadata.
        """
        return ExplanationMetadata(
            explainer_name=self.get_explainer_name(),
            method=self.get_method_name(),
            model_name=model_name,
            framework=self.get_framework_name(),
            execution_time_ms=execution_time_ms,
            additional_metadata=additional_metadata or {},
        )

    @abstractmethod
    def explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> ExplanationResult:
        """
        Generate an explanation for an agent result.
        """
        raise NotImplementedError