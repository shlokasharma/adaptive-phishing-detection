"""
Email Analysis Agent.

This module connects the Phase 2 email baseline detector
with the Phase 3 agent architecture.

The agent uses:
    - Phase 2 TF-IDF feature extractor
    - Phase 2 Linear SVM classifier

It produces:
    - standardized phishing prediction
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

from phishing_detection.agents.base import (
    AgentContext,
    AgentExecutionMetadata,
    BaseAgent,
)
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult


class EmailAnalysisAgent(BaseAgent):
    """
    Specialized agent for email-content phishing analysis.

    The agent reuses the frozen Phase 2 Linear SVM model and
    TF-IDF feature extractor.

    Parameters
    ----------
    model_path:
        Path to the trained Phase 2 email classifier.

    extractor_path:
        Path to the trained Phase 2 TF-IDF extractor.

    agent_name:
        Name used to identify this agent in the orchestration layer.
    """

    DEFAULT_AGENT_NAME = "email_analysis_agent"
    DEFAULT_MODALITY = "email"

    def __init__(
        self,
        model_path: str | Path,
        extractor_path: str | Path,
        agent_name: str = DEFAULT_AGENT_NAME,
    ) -> None:
        self.model_path = Path(model_path)
        self.extractor_path = Path(extractor_path)
        self.agent_name = agent_name

        self._model: Any | None = None
        self._extractor: Any | None = None

    def get_agent_name(self) -> str:
        """Return the unique agent name."""
        return self.agent_name

    def get_modality(self) -> str:
        """Return the modality analyzed by this agent."""
        return self.DEFAULT_MODALITY

    def get_capabilities(self) -> list[str]:
        """Return capabilities provided by the agent."""
        return [
            "email_text_analysis",
            "tfidf_feature_extraction",
            "linear_svm_classification",
            "phishing_probability_estimation",
        ]

    def can_analyze(self, context: AgentContext) -> bool:
        """
        Determine whether the agent can analyze the supplied context.

        Email analysis requires non-empty input text.
        """
        return bool(
            context.input_text
            and context.input_text.strip()
        )

    def _load_artifacts(self) -> None:
        """
        Load the Phase 2 model and TF-IDF extractor.

        Artifacts are loaded lazily so importing the agent does not
        immediately load potentially large model files.
        """

        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "Email model artifact was not found: "
                    f"{self.model_path}"
                )

            self._model = joblib.load(self.model_path)

        if self._extractor is None:
            if not self.extractor_path.exists():
                raise FileNotFoundError(
                    "Email TF-IDF extractor artifact was not found: "
                    f"{self.extractor_path}"
                )

            self._extractor = joblib.load(self.extractor_path)

    def analyze(self, context: AgentContext) -> AgentResult:
        """
        Analyze email text and return a standardized AgentResult.

        Parameters
        ----------
        context:
            AgentContext containing the email text.

        Returns
        -------
        AgentResult
            Standardized result containing prediction and evidence.
        """

        self.validate_context(context)

        if not self.can_analyze(context):
            raise ValueError(
                "EmailAnalysisAgent requires non-empty input_text."
            )

        start_time = time.perf_counter()

        self._load_artifacts()

        transformed_text = self._extractor.transform(
            [context.input_text]
        )

        prediction = int(
            self._model.predict(transformed_text)[0]
        )

        probability = float(
            self._model.predict_proba(transformed_text)[0][1]
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

        polarity = (
            "supports_phishing"
            if prediction == 1
            else "supports_legitimate"
        )

        label_name = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        evidence = Evidence(
            evidence_type="model_prediction",
            source=self.agent_name,
            description=(
                "Phase 2 Linear SVM analysis of email text "
                "using the frozen TF-IDF feature representation."
            ),
            value={
                "prediction": label_name,
                "phishing_probability": probability,
            },
            polarity=polarity,
            confidence=confidence,
            reliability=1.0,
            metadata={
                "model": "email_linear_svm",
                "feature_extractor": "email_tfidf",
            },
        )

        execution_time_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        execution_metadata = AgentExecutionMetadata(
            execution_time_ms=execution_time_ms,
            model_name="email_linear_svm",
            additional_metadata={
                "model_path": str(self.model_path),
                "extractor_path": str(self.extractor_path),
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
                "analysis_type": "email_content",
                "phase": "3.4",
            },
        )