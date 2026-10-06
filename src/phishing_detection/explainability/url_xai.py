"""
URL explainability using SHAP and LIME.

Phase 4.4
---------
Provides:
    - URLSHAPExplainer
    - URLLIMEExplainer
    - URLXAIExplainer

The explainers operate on the frozen Phase 2 URL Random Forest
model and its corresponding feature extractor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "url_random_forest.pkl"
)

DEFAULT_FEATURE_EXTRACTOR_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "url_random_forest_feature_extractor.pkl"
)

DEFAULT_BACKGROUND_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "train.csv"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _prediction_target(result: AgentResult) -> ExplanationTarget:
    """Create the unified explanation target from an AgentResult."""

    return ExplanationTarget(
        label=result.label,
        label_name=result.label_name,
        probability=float(result.phishing_probability),
    )


def _direction_from_contribution(
    contribution: float,
) -> ExplanationDirection:
    """Map a signed model contribution to an explanation direction."""

    if contribution > 0:
        return ExplanationDirection.SUPPORTS_PHISHING

    if contribution < 0:
        return ExplanationDirection.SUPPORTS_LEGITIMATE

    return ExplanationDirection.NEUTRAL


def _get_url_features(
    context: AgentContext,
) -> Any:
    """Extract URL feature representation from AgentContext."""

    if context is None:
        raise ValueError("context cannot be None")

    metadata = context.metadata or {}

    if "url_features" not in metadata:
        raise ValueError(
            "AgentContext.metadata['url_features'] is required"
        )

    return metadata["url_features"]


def _to_dataframe_like(
    features: Any,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert URL feature input into a one-row DataFrame."""

    if isinstance(features, pd.DataFrame):
        dataframe = features.copy()

    elif isinstance(features, dict):
        dataframe = pd.DataFrame([features])

    elif isinstance(features, (list, tuple, np.ndarray)):
        array = np.asarray(features)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                "URL features must be one- or two-dimensional"
            )

        if feature_names is not None:
            if array.shape[1] != len(feature_names):
                raise ValueError(
                    "Number of URL features does not match "
                    "the number of feature names"
                )

            dataframe = pd.DataFrame(
                array,
                columns=feature_names,
            )

        else:
            dataframe = pd.DataFrame(array)

    else:
        raise TypeError(
            "Unsupported URL feature representation: "
            f"{type(features).__name__}"
        )

    return dataframe


def _get_feature_names(
    feature_extractor: Any,
) -> list[str]:
    """Obtain feature names from the frozen feature extractor."""

    if feature_extractor is None:
        return []

    if hasattr(feature_extractor, "feature_names"):
        names = getattr(
            feature_extractor,
            "feature_names",
        )

        if names is not None:
            return [str(name) for name in names]

    if hasattr(feature_extractor, "feature_names_in_"):
        names = getattr(
            feature_extractor,
            "feature_names_in_",
        )

        if names is not None:
            return [str(name) for name in names]

    if hasattr(feature_extractor, "get_feature_names_out"):
        try:
            names = feature_extractor.get_feature_names_out()

            if names is not None:
                return [str(name) for name in names]

        except Exception:
            pass

    return []


def _transform_features(
    feature_extractor: Any,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Apply the frozen Phase 2 URL feature preprocessing."""

    if feature_extractor is None:
        return dataframe.to_numpy(dtype=float)

    if hasattr(feature_extractor, "transform"):
        transformed = feature_extractor.transform(dataframe)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        return np.asarray(
            transformed,
            dtype=float,
        )

    return dataframe.to_numpy(dtype=float)


def _model_probability(
    model: Any,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Return phishing probabilities from a binary URL classifier.

    Project convention:
        class 0 = legitimate
        class 1 = phishing

    Therefore, column 1 of predict_proba() is the phishing
    probability.
    """

    if model is None:
        raise ValueError(
            "model cannot be None"
        )

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "URL model does not expose predict_proba()"
        )

    feature_matrix = np.asarray(
        feature_matrix,
        dtype=float,
    )

    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(
            1,
            -1,
        )

    if feature_matrix.ndim != 2:
        raise ValueError(
            "feature_matrix must be one- or two-dimensional"
        )

    probabilities = model.predict_proba(
        feature_matrix
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected predict_proba() to return a 2D array"
        )

    if probabilities.shape[1] < 2:
        raise ValueError(
            "Expected binary classification probabilities "
            "with at least two columns"
        )

    return probabilities[:, 1]


def _find_tree_estimator(
    model: Any,
) -> Any:
    """
    Extract the underlying sklearn tree ensemble.

    The frozen Phase 2 artifact is a custom
    URLRandomForestDetector wrapper. SHAP TreeExplainer cannot
    explain that wrapper directly, so this function recursively
    searches for the fitted sklearn estimator.
    """

    if model is None:
        raise ValueError(
            "model cannot be None"
        )

    # Direct sklearn estimator.
    if hasattr(model, "estimators_") and hasattr(
        model,
        "predict_proba",
    ):
        return model

    # Common wrapper attributes.
    candidate_attributes = (
        "model",
        "classifier",
        "estimator",
        "random_forest",
        "rf_model",
        "_model",
        "_classifier",
        "_estimator",
    )

    for attribute in candidate_attributes:
        if hasattr(model, attribute):
            candidate = getattr(
                model,
                attribute,
            )

            if candidate is model:
                continue

            try:
                return _find_tree_estimator(
                    candidate
                )
            except ValueError:
                continue

    raise ValueError(
        "Could not locate the underlying fitted tree "
        "ensemble inside the URL model artifact"
    )


def _extract_shap_values(
    shap_values: Any,
) -> np.ndarray:
    """Normalize SHAP output into a one-dimensional vector."""

    values = shap_values

    if hasattr(values, "values"):
        values = values.values

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim == 3:
        # Binary classification can appear as:
        # samples x features x classes
        #
        # Class 1 corresponds to phishing.
        if values.shape[-1] >= 2:
            values = values[0, :, 1]
        else:
            values = values[0, :, 0]

    elif values.ndim == 2:
        values = values[0]

    elif values.ndim != 1:
        values = values.reshape(-1)

    return values


# ---------------------------------------------------------------------------
# URL SHAP Explainer
# ---------------------------------------------------------------------------

class URLSHAPExplainer:
    """SHAP-based URL explanation."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        feature_extractor_path: str | Path = DEFAULT_FEATURE_EXTRACTOR_PATH,
        max_display: int = 15,
    ) -> None:

        if max_display <= 0:
            raise ValueError(
                "max_display must be greater than 0"
            )

        self.model_path = Path(model_path)
        self.feature_extractor_path = Path(
            feature_extractor_path
        )
        self.max_display = int(max_display)

        # Public metadata expected by the Phase 4 tests.
        self.explainer_name = "url_shap_explainer"
        self.method = "shap"
        self.method_name = self.method
        self.framework = "SHAP"
        self.framework_name = "shap"

        self._model: Any | None = None
        self._tree_model: Any | None = None
        self._feature_extractor: Any | None = None
        self._explainer: Any | None = None

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_explainer_name(self) -> str:
        return self.explainer_name

    def get_method_name(self) -> str:
        return self.method

    def get_framework_name(self) -> str:
        return self.framework_name

    def can_explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> bool:
        """Return True only for URL analysis results."""

        if result is None or context is None:
            return False

        return (
            result.modality == "url"
            and context.url is not None
            and context.metadata is not None
            and "url_features" in context.metadata
        )

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
                "URLSHAPExplainer requires a URL AgentResult "
                "and URL features in AgentContext.metadata"
            )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:

        if self._model is None:

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"URL model not found: {self.model_path}"
                )

            self._model = joblib.load(
                self.model_path
            )

        if self._feature_extractor is None:

            if not self.feature_extractor_path.exists():
                raise FileNotFoundError(
                    "URL feature extractor not found: "
                    f"{self.feature_extractor_path}"
                )

            self._feature_extractor = joblib.load(
                self.feature_extractor_path
            )

        if self._tree_model is None:
            self._tree_model = _find_tree_estimator(
                self._model
            )

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(
                self._tree_model
            )

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

        self._load()

        raw_features = _get_url_features(
            context
        )

        feature_names = _get_feature_names(
            self._feature_extractor
        )

        dataframe = _to_dataframe_like(
            raw_features,
            feature_names if feature_names else None,
        )

        transformed = _transform_features(
            self._feature_extractor,
            dataframe,
        )

        shap_values = self._explainer.shap_values(
            transformed
        )

        contributions = _extract_shap_values(
            shap_values
        )

        if not feature_names:
            feature_names = [
                f"feature_{index}"
                for index in range(
                    len(contributions)
                )
            ]

        if len(feature_names) != len(contributions):
            feature_names = [
                f"feature_{index}"
                for index in range(
                    len(contributions)
                )
            ]

        rows: list[tuple[int, str, float, float]] = []

        for index, (
            feature,
            contribution,
        ) in enumerate(
            zip(
                feature_names,
                contributions,
            )
        ):
            rows.append(
                (
                    index,
                    str(feature),
                    float(dataframe.iloc[0, index])
                    if index < dataframe.shape[1]
                    else float("nan"),
                    float(contribution),
                )
            )

        rows.sort(
            key=lambda item: abs(item[3]),
            reverse=True,
        )

        selected_rows = rows[
            : self.max_display
        ]

        factors: list[ExplanationFactor] = []

        for rank, (
            index,
            feature,
            value,
            contribution,
        ) in enumerate(
            selected_rows,
            start=1,
        ):

            factors.append(
                ExplanationFactor(
                    feature=feature,
                    value=value,
                    importance=abs(
                        contribution
                    ),
                    direction=(
                        _direction_from_contribution(
                            contribution
                        )
                    ),
                    rank=rank,
                    contribution=contribution,
                    metadata={
                        "feature_index": index,
                        "explainer": "SHAP",
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

        summary = (
            "SHAP identified "
            f"{phishing_count} feature(s) supporting phishing "
            f"and {legitimate_count} feature(s) supporting "
            "legitimate classification."
        )

        execution_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        metadata = ExplanationMetadata(
            explainer_name=self.explainer_name,
            method=self.method,
            model_name="url_random_forest",
            framework=self.framework,
            execution_time_ms=execution_time_ms,
            additional_metadata={
                "phase": "4.4",
                "feature_count": len(
                    feature_names
                ),
                "displayed_feature_count": len(
                    factors
                ),
            },
        )

        return UnifiedExplanation(
            summary=summary,
            explainer_name=self.explainer_name,
            method=self.method,
            framework=self.framework,
            scope=ExplanationScope.LOCAL,
            target=_prediction_target(result),
            factors=factors,
            model_name="url_random_forest",
            execution_time_ms=execution_time_ms,
            metadata=metadata.to_dict(),
        )


# ---------------------------------------------------------------------------
# URL LIME Explainer
# ---------------------------------------------------------------------------

class URLLIMEExplainer:
    """LIME-based URL explanation."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        feature_extractor_path: str | Path = DEFAULT_FEATURE_EXTRACTOR_PATH,
        background_path: str | Path = DEFAULT_BACKGROUND_PATH,
        num_features: int = 15,
        num_samples: int = 1000,
        background_size: int = 500,
    ) -> None:

        if num_features <= 0:
            raise ValueError(
                "num_features must be greater than 0"
            )

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than 0"
            )

        if background_size <= 0:
            raise ValueError(
                "background_size must be greater than 0"
            )

        self.model_path = Path(model_path)
        self.feature_extractor_path = Path(
            feature_extractor_path
        )
        self.background_path = Path(
            background_path
        )

        self.num_features = int(
            num_features
        )
        self.num_samples = int(
            num_samples
        )
        self.background_size = int(
            background_size
        )

        # Public metadata expected by the Phase 4 tests.
        self.explainer_name = "url_lime_explainer"
        self.method = "lime"
        self.method_name = self.method
        self.framework = "LIME"
        self.framework_name = "lime"

        self._model: Any | None = None
        self._tree_model: Any | None = None
        self._feature_extractor: Any | None = None
        self._explainer: Any | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_explainer_name(self) -> str:
        return self.explainer_name

    def get_method_name(self) -> str:
        return self.method

    def get_framework_name(self) -> str:
        return self.framework_name

    def can_explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> bool:
        """Return True only for URL analysis results."""

        if result is None or context is None:
            return False

        return (
            result.modality == "url"
            and context.url is not None
            and context.metadata is not None
            and "url_features" in context.metadata
        )

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
                "URLLIMEExplainer requires a URL AgentResult "
                "and URL features in AgentContext.metadata"
            )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:

        if self._model is None:

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"URL model not found: {self.model_path}"
                )

            self._model = joblib.load(
                self.model_path
            )

        if self._tree_model is None:
            self._tree_model = _find_tree_estimator(
                self._model
            )

        if self._feature_extractor is None:

            if not self.feature_extractor_path.exists():
                raise FileNotFoundError(
                    "URL feature extractor not found: "
                    f"{self.feature_extractor_path}"
                )

            self._feature_extractor = joblib.load(
                self.feature_extractor_path
            )

        if self._explainer is not None:
            return

        if not self.background_path.exists():
            raise FileNotFoundError(
                "LIME background dataset not found: "
                f"{self.background_path}"
            )

        background = pd.read_csv(
            self.background_path
        )

        drop_columns = [
            "sample_id",
            "source_dataset",
            "label",
        ]

        background = background.drop(
            columns=[
                column
                for column in drop_columns
                if column in background.columns
            ],
            errors="ignore",
        )

        self._feature_names = [
            str(column)
            for column in background.columns
        ]

        if len(background) > self.background_size:
            background = background.sample(
                n=self.background_size,
                random_state=42,
            )

        background_transformed = (
            _transform_features(
                self._feature_extractor,
                background,
            )
        )

        self._explainer = LimeTabularExplainer(
            training_data=background_transformed,
            feature_names=self._feature_names,
            class_names=[
                "legitimate",
                "phishing",
            ],
            mode="classification",
            discretize_continuous=False,
            random_state=42,
        )

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

        self._load()

        raw_features = _get_url_features(
            context
        )

        dataframe = _to_dataframe_like(
            raw_features,
            self._feature_names
            if self._feature_names
            else None,
        )

        transformed = _transform_features(
            self._feature_extractor,
            dataframe,
        )

        instance = transformed[0]

        def predict_proba(
            instances: np.ndarray,
        ) -> np.ndarray:
            return np.column_stack(
                [
                    1.0
                    - _model_probability(
                        self._tree_model,
                        instances,
                    ),
                    _model_probability(
                        self._tree_model,
                        instances,
                    ),
                ]
            )

        explanation = self._explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_proba,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=(1,),
        )

        lime_items = explanation.as_list(
            label=1
        )

        factors: list[ExplanationFactor] = []

        for rank, (
            feature,
            contribution,
        ) in enumerate(
            lime_items,
            start=1,
        ):

            contribution = float(
                contribution
            )

            factors.append(
                ExplanationFactor(
                    feature=str(feature),
                    value=None,
                    importance=abs(
                        contribution
                    ),
                    direction=(
                        _direction_from_contribution(
                            contribution
                        )
                    ),
                    rank=rank,
                    contribution=contribution,
                    metadata={
                        "explainer": "LIME",
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

        summary = (
            "LIME identified "
            f"{phishing_count} feature(s) supporting phishing "
            f"and {legitimate_count} feature(s) supporting "
            "legitimate classification."
        )

        execution_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        metadata = ExplanationMetadata(
            explainer_name=self.explainer_name,
            method=self.method,
            model_name="url_random_forest",
            framework=self.framework,
            execution_time_ms=execution_time_ms,
            additional_metadata={
                "phase": "4.4",
                "background_size": self.background_size,
                "num_samples": self.num_samples,
                "displayed_feature_count": len(
                    factors
                ),
            },
        )

        return UnifiedExplanation(
            summary=summary,
            explainer_name=self.explainer_name,
            method=self.method,
            framework=self.framework,
            scope=ExplanationScope.LOCAL,
            target=_prediction_target(result),
            factors=factors,
            model_name="url_random_forest",
            execution_time_ms=execution_time_ms,
            metadata=metadata.to_dict(),
        )


# ---------------------------------------------------------------------------
# Combined URL XAI Explainer
# ---------------------------------------------------------------------------

class URLXAIExplainer:
    """Simple combined SHAP + LIME URL explainer."""

    def __init__(
        self,
        shap_explainer=None,
        lime_explainer=None,
    ):
        # Individual explainers
        self.shap = (
            shap_explainer
            if shap_explainer is not None
            else URLSHAPExplainer()
        )

        self.lime = (
            lime_explainer
            if lime_explainer is not None
            else URLLIMEExplainer()
        )

        # Backward-compatible descriptive names
        self.shap_explainer = self.shap
        self.lime_explainer = self.lime

        # Public metadata
        self.explainer_name = "url_xai_explainer"
        self.method = "combined"
        self.framework = "SHAP+LIME"
        self.framework_name = "shap+lime"

    def get_explainer_name(self):
        return self.explainer_name

    def get_method_name(self):
        return self.method

    def get_framework_name(self):
        return self.framework_name

    def can_explain(self, result, context):
        return (
            self.shap.can_explain(
                result,
                context,
            )
            and
            self.lime.can_explain(
                result,
                context,
            )
        )

    def validate_inputs(self, result, context):
        if not self.can_explain(
            result,
            context,
        ):
            raise ValueError(
                "URLXAIExplainer requires a valid URL "
                "AgentResult and URL features."
            )

    def explain(self, result, context):
        self.validate_inputs(
            result,
            context,
        )

        return {
            "shap": self.shap.explain(
                result,
                context,
            ),
            "lime": self.lime.explain(
                result,
                context,
            ),
        }

    
# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "URLSHAPExplainer",
    "URLLIMEExplainer",
    "URLXAIExplainer",
    "_model_probability",
]