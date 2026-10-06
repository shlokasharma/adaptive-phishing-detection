"""
Feature-based Random Forest phishing URL detector.

This module implements a Phase 2 traditional machine-learning
baseline for phishing URL detection.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from phishing_detection.models.base import (
    BasePhishingDetector,
    PredictionResult,
)


class URLRandomForestDetector(BasePhishingDetector):
    """
    Random Forest detector for numerical URL features.

    URL preprocessing is handled separately by URLFeatureExtractor.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: Iterable[int],
    ) -> "URLRandomForestDetector":
        """
        Fit the Random Forest model.
        """

        self.model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict URL labels.

        Returns
        -------
        numpy.ndarray
            0 = legitimate
            1 = phishing
        """

        self._check_fitted()

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.
        """

        self._check_fitted()

        return self.model.predict_proba(X)

    def predict_one(self, X: np.ndarray) -> PredictionResult:
        """
        Generate a standardized prediction for one URL sample.
        """

        self._check_fitted()

        if X.shape[0] != 1:
            raise ValueError(
                "predict_one() expects exactly one sample."
            )

        probabilities = self.predict_proba(X)

        phishing_probability = float(
            probabilities[0, 1]
        )

        label = int(
            self.predict(X)[0]
        )

        return self.build_prediction_result(
            label=label,
            phishing_probability=phishing_probability,
            metadata={
                "model_type": "random_forest",
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
            },
        )

    def get_model_name(self) -> str:
        """
        Return the unique detector name.
        """

        return "url_random_forest"

    def get_modality(self) -> str:
        """
        Return the evidence modality.
        """

        return "url"

    def _check_fitted(self) -> None:
        """
        Ensure the detector has been trained before inference.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "URLRandomForestDetector must be fitted "
                "before prediction."
            )