"""
SHAP and LIME explainers for the Phase 2 email detector.

The explainers operate on raw email text and use the frozen Phase 2
TF-IDF + calibrated Linear SVM artifacts.

No model retraining is performed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.base import BaseExplainer
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


DEFAULT_MODEL_PATH = (
    Path("experiments")
    / "baselines"
    / "models"
    / "email_linear_svm.pkl"
)

DEFAULT_TFIDF_PATH = (
    Path("experiments")
    / "baselines"
    / "models"
    / "email_linear_svm_tfidf_extractor.pkl"
)


def _get_email_text(context: AgentContext) -> str:
    """
    Extract the email text used for explanation.
    """
    if not context.input_text:
        raise ValueError(
            "AgentContext.input_text is required for email explanation."
        )

    text = context.input_text.strip()

    if not text:
        raise ValueError(
            "AgentContext.input_text must not be empty."
        )

    return text


def _prediction_target(result: AgentResult) -> ExplanationTarget:
    """
    Convert an AgentResult into the standardized explanation target.
    """
    return ExplanationTarget(
        label=result.label,
        label_name=result.label_name,
        probability=result.phishing_probability,
    )


def _direction_from_contribution(
    contribution: float,
) -> ExplanationDirection:
    """
    Convert a signed contribution into an explanation direction.
    """
    if contribution > 0:
        return ExplanationDirection.SUPPORTS_PHISHING

    if contribution < 0:
        return ExplanationDirection.SUPPORTS_LEGITIMATE

    return ExplanationDirection.NEUTRAL


def _model_probability(
    model: Any,
    vectorizer: Any,
    texts: list[str],
) -> np.ndarray:
    """
    Return phishing probabilities for raw email texts.

    The saved Phase 2 TF-IDF extractor is reused exactly as trained.
    """
    transformed = vectorizer.transform(texts)
    probabilities = model.predict_proba(transformed)

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Email model must provide a two-class predict_proba output."
        )

    return probabilities[:, 1]


class EmailSHAPExplainer(BaseExplainer):
    """
    SHAP-based local explainer for the Phase 2 email detector.

    SHAP operates on raw email text and internally evaluates the exact
    frozen email prediction function.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        tfidf_path: str | Path = DEFAULT_TFIDF_PATH,
        max_evals: int = 200,
        max_display: int = 15,
    ) -> None:
        if max_evals < 1:
            raise ValueError("max_evals must be positive.")

        if max_display < 1:
            raise ValueError("max_display must be positive.")

        self.model_path = Path(model_path)
        self.tfidf_path = Path(tfidf_path)
        self.max_evals = max_evals
        self.max_display = max_display

        self._model: Any | None = None
        self._vectorizer: Any | None = None
        self._explainer: Any | None = None

    def get_explainer_name(self) -> str:
        return "email_shap_explainer"

    def get_method_name(self) -> str:
        return "shap"

    def get_framework_name(self) -> str:
        return "SHAP"

    def _load_artifacts(self) -> None:
        """
        Lazily load the frozen Phase 2 artifacts.
        """
        if self._model is None:
            self._model = joblib.load(self.model_path)

        if self._vectorizer is None:
            self._vectorizer = joblib.load(self.tfidf_path)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Prediction function supplied to SHAP.
        """
        self._load_artifacts()

        return _model_probability(
            self._model,
            self._vectorizer,
            texts,
        )

    def _build_explainer(self) -> Any:
        """
        Lazily construct the SHAP text explainer.
        """
        if self._explainer is None:
            masker = shap.maskers.Text(
                tokenizer=None,
            )

            self._explainer = shap.Explainer(
                self._predict_proba,
                masker,
                algorithm="partition",
            )

        return self._explainer

    def explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> UnifiedExplanation:
        """
        Generate a local SHAP explanation.
        """
        self.validate_inputs(result, context)

        text = _get_email_text(context)

        start_time = time.perf_counter()

        explainer = self._build_explainer()

        shap_result = explainer(
            [text],
            max_evals=self.max_evals,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        explanation = shap_result[0]

        tokens = list(explanation.data)

        values = np.asarray(explanation.values)

        if values.ndim > 1:
            values = np.squeeze(values)

        values = values.reshape(-1)

        factor_count = min(
            len(tokens),
            len(values),
            self.max_display,
        )

        pairs = []

        for index in range(factor_count):
            token = str(tokens[index])
            contribution = float(values[index])

            if not token.strip():
                continue

            pairs.append(
                (
                    token,
                    contribution,
                )
            )

        pairs.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        factors: list[ExplanationFactor] = []

        for rank, (token, contribution) in enumerate(
            pairs[: self.max_display],
            start=1,
        ):
            factors.append(
                ExplanationFactor(
                    feature=token,
                    value=token,
                    importance=abs(contribution),
                    direction=_direction_from_contribution(
                        contribution
                    ),
                    rank=rank,
                    contribution=contribution,
                    metadata={
                        "source": "shap",
                        "modality": "email",
                    },
                )
            )

        predicted_label = result.label_name

        summary = (
            f"SHAP identified {len(factors)} influential "
            f"email tokens for the {predicted_label} prediction."
        )

        metadata = self.create_metadata(
            model_name="email_linear_svm",
            execution_time_ms=elapsed_ms,
            additional_metadata={
                "phase": "4.3",
                "max_evals": self.max_evals,
                "max_display": self.max_display,
            },
        )

        return UnifiedExplanation(
            summary=summary,
            explainer_name=self.get_explainer_name(),
            method=self.get_method_name(),
            framework=self.get_framework_name(),
            scope=ExplanationScope.LOCAL,
            target=_prediction_target(result),
            factors=factors,
            model_name="email_linear_svm",
            execution_time_ms=elapsed_ms,
            metadata=metadata.to_dict(),
        )


class EmailLIMEExplainer(BaseExplainer):
    """
    LIME-based local explainer for the Phase 2 email detector.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        tfidf_path: str | Path = DEFAULT_TFIDF_PATH,
        num_features: int = 15,
        num_samples: int = 1000,
        class_names: list[str] | None = None,
    ) -> None:
        if num_features < 1:
            raise ValueError(
                "num_features must be positive."
            )

        if num_samples < 1:
            raise ValueError(
                "num_samples must be positive."
            )

        self.model_path = Path(model_path)
        self.tfidf_path = Path(tfidf_path)
        self.num_features = num_features
        self.num_samples = num_samples

        self.class_names = class_names or [
            "legitimate",
            "phishing",
        ]

        if len(self.class_names) != 2:
            raise ValueError(
                "class_names must contain exactly two classes."
            )

        self._model: Any | None = None
        self._vectorizer: Any | None = None
        self._explainer: LimeTextExplainer | None = None

    def get_explainer_name(self) -> str:
        return "email_lime_explainer"

    def get_method_name(self) -> str:
        return "lime"

    def get_framework_name(self) -> str:
        return "LIME"

    def _load_artifacts(self) -> None:
        """
        Lazily load the frozen Phase 2 artifacts.
        """
        if self._model is None:
            self._model = joblib.load(self.model_path)

        if self._vectorizer is None:
            self._vectorizer = joblib.load(self.tfidf_path)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Prediction function supplied to LIME.
        """
        self._load_artifacts()

        transformed = self._vectorizer.transform(texts)

        probabilities = self._model.predict_proba(
            transformed
        )

        return np.asarray(probabilities)

    def _build_explainer(self) -> LimeTextExplainer:
        """
        Lazily construct the LIME text explainer.
        """
        if self._explainer is None:
            self._explainer = LimeTextExplainer(
                class_names=self.class_names,
                random_state=42,
            )

        return self._explainer

    def explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> UnifiedExplanation:
        """
        Generate a local LIME explanation.
        """
        self.validate_inputs(result, context)

        text = _get_email_text(context)

        start_time = time.perf_counter()

        explainer = self._build_explainer()

        lime_result = explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=(1,),
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        local_list = lime_result.as_list(
            label=1,
        )

        factors: list[ExplanationFactor] = []

        sorted_items = sorted(
            local_list,
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )

        for rank, (token, contribution) in enumerate(
            sorted_items[: self.num_features],
            start=1,
        ):
            contribution = float(contribution)

            factors.append(
                ExplanationFactor(
                    feature=str(token),
                    value=str(token),
                    importance=abs(contribution),
                    direction=_direction_from_contribution(
                        contribution
                    ),
                    rank=rank,
                    contribution=contribution,
                    metadata={
                        "source": "lime",
                        "modality": "email",
                    },
                )
            )

        summary = (
            f"LIME identified {len(factors)} influential "
            f"email tokens for the {result.label_name} prediction."
        )

        metadata = self.create_metadata(
            model_name="email_linear_svm",
            execution_time_ms=elapsed_ms,
            additional_metadata={
                "phase": "4.3",
                "num_features": self.num_features,
                "num_samples": self.num_samples,
                "random_state": 42,
            },
        )

        return UnifiedExplanation(
            summary=summary,
            explainer_name=self.get_explainer_name(),
            method=self.get_method_name(),
            framework=self.get_framework_name(),
            scope=ExplanationScope.LOCAL,
            target=_prediction_target(result),
            factors=factors,
            model_name="email_linear_svm",
            execution_time_ms=elapsed_ms,
            metadata=metadata.to_dict(),
        )


class EmailXAIExplainer:
    """
    Convenience wrapper providing both SHAP and LIME explanations.

    This class does not replace the individual explainers. It provides
    a single interface for generating both explanations for the same
    email prediction.
    """

    def __init__(
        self,
        shap_explainer: EmailSHAPExplainer | None = None,
        lime_explainer: EmailLIMEExplainer | None = None,
    ) -> None:
        self.shap_explainer = (
            shap_explainer
            or EmailSHAPExplainer()
        )

        self.lime_explainer = (
            lime_explainer
            or EmailLIMEExplainer()
        )

    def explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> dict[str, UnifiedExplanation]:
        """
        Generate both SHAP and LIME explanations.
        """
        return {
            "shap": self.shap_explainer.explain(
                result,
                context,
            ),
            "lime": self.lime_explainer.explain(
                result,
                context,
            ),
        }