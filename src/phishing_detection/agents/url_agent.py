"""
URL Analysis Agent.

This module connects the Phase 2 URL baseline detector with the
Phase 3 agent architecture.

The agent operates on the engineered URL feature representation
used by the Phase 2 UCI phishing-website baseline.

Expected input:

    AgentContext(
        url="https://example.com",
        metadata={
            "url_features": {
                ...
            }
        }
    )

The agent produces:

    - phishing prediction
    - phishing probability
    - confidence
    - uncertainty
    - structured evidence
    - execution metadata

The agent does not make the final system-level decision.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from phishing_detection.agents.base import (
    AgentContext,
    AgentExecutionMetadata,
    BaseAgent,
)
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult


class URLAnalysisAgent(BaseAgent):
    """
    Specialized agent for URL-based phishing analysis.

    Parameters
    ----------
    model_path:
        Path to the trained Phase 2 URL classifier.

    preprocessor_path:
        Path to the Phase 2 URL feature preprocessor.

    agent_name:
        Name used to identify this agent.
    """

    DEFAULT_AGENT_NAME = "url_analysis_agent"
    DEFAULT_MODALITY = "url"

    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
        agent_name: str = DEFAULT_AGENT_NAME,
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.agent_name = agent_name

        self._model: Any | None = None
        self._preprocessor: Any | None = None

    def get_agent_name(self) -> str:
        """Return the unique agent name."""
        return self.agent_name

    def get_modality(self) -> str:
        """Return the modality analyzed by this agent."""
        return self.DEFAULT_MODALITY

    def get_capabilities(self) -> list[str]:
        """Return capabilities provided by the agent."""
        return [
            "url_analysis",
            "engineered_url_feature_analysis",
            "random_forest_classification",
            "phishing_probability_estimation",
        ]

    def can_analyze(self, context: AgentContext) -> bool:
        """
        Determine whether this agent can analyze the context.

        URL analysis requires:

        1. A non-empty URL.
        2. A ``url_features`` object in context metadata.
        """

        if not context.url or not context.url.strip():
            return False

        if not isinstance(context.metadata, dict):
            return False

        url_features = context.metadata.get("url_features")

        return url_features is not None

    def _load_artifacts(self) -> None:
        """Load the Phase 2 URL model and preprocessor lazily."""

        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "URL model artifact was not found: "
                    f"{self.model_path}"
                )

            self._model = joblib.load(self.model_path)

        if self._preprocessor is None:
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(
                    "URL preprocessor artifact was not found: "
                    f"{self.preprocessor_path}"
                )

            self._preprocessor = joblib.load(
                self.preprocessor_path
            )

    @staticmethod
    def _prepare_features(
        url_features: Any,
    ) -> np.ndarray:
        """
        Convert URL feature input into a two-dimensional array.

        The Phase 2 URL baseline expects a numeric feature matrix.

        A dictionary is converted using insertion order. Therefore,
        production code should provide the feature dictionary in the
        same order as the frozen Phase 2 feature schema.
        """

        if isinstance(url_features, dict):
            values = list(url_features.values())
        elif isinstance(url_features, (list, tuple, np.ndarray)):
            values = list(url_features)
        else:
            raise TypeError(
                "url_features must be a dictionary, list, tuple, "
                "or numpy array."
            )

        if not values:
            raise ValueError(
                "url_features must contain at least one feature."
            )

        try:
            numeric_values = [
                float(value)
                for value in values
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All URL feature values must be numeric."
            ) from exc

        return np.asarray(
            [numeric_values],
            dtype=float,
        )

    def analyze(self, context: AgentContext) -> AgentResult:
        """
        Analyze URL features and return a standardized AgentResult.

        Parameters
        ----------
        context:
            AgentContext containing the URL and engineered URL
            features.

        Returns
        -------
        AgentResult
            Standardized URL analysis result.
        """

        self.validate_context(context)

        if not self.can_analyze(context):
            raise ValueError(
                "URLAnalysisAgent requires a non-empty URL and "
                "metadata['url_features']."
            )

        start_time = time.perf_counter()

        self._load_artifacts()

        url_features = context.metadata["url_features"]

        feature_matrix = self._prepare_features(
            url_features
        )

        transformed_features = self._preprocessor.transform(
            feature_matrix
        )

        prediction = int(
            self._model.predict(transformed_features)[0]
        )

        probability = float(
            self._model.predict_proba(
                transformed_features
            )[0][1]
        )

        confidence = (
            probability
            if prediction == 1
            else 1.0 - probability
        )

        uncertainty = round(
            1.0 - confidence,
            10,
        )

        label_name = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        polarity = (
            "supports_phishing"
            if prediction == 1
            else "supports_legitimate"
        )

        evidence = Evidence(
            evidence_type="model_prediction",
            source=self.agent_name,
            description=(
                "Phase 2 Random Forest analysis of the engineered "
                "URL feature representation."
            ),
            value={
                "prediction": label_name,
                "phishing_probability": probability,
                "url": context.url,
            },
            polarity=polarity,
            confidence=confidence,
            reliability=1.0,
            metadata={
                "model": "url_random_forest",
                "feature_representation": (
                    "uci_phishing_website_features"
                ),
                "feature_count": int(feature_matrix.shape[1]),
            },
        )

        execution_time_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        execution_metadata = AgentExecutionMetadata(
            execution_time_ms=execution_time_ms,
            model_name="url_random_forest",
            additional_metadata={
                "model_path": str(self.model_path),
                "preprocessor_path": str(
                    self.preprocessor_path
                ),
                "feature_count": int(
                    feature_matrix.shape[1]
                ),
            },
        )

        return AgentResult(
            agent_name=self.agent_name,
            modality=self.DEFAULT_MODALITY,
            label=prediction,
            label_name=label_name,
            phishing_probability=probability,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=[evidence],
            execution_metadata=execution_metadata,
            metadata={
                "analysis_type": "url_features",
                "phase": "3.5",
            },
        )