"""
Feature-based Gradient Boosting phishing URL detector.

This module implements a Phase 2 traditional machine-learning
baseline for phishing URL detection.

Pipeline:

    Numerical URL features
        ↓
    Gradient Boosting Classifier
        ↓
    Phishing probability
        ↓
    Standardized PredictionResult
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from phishing_detection.models.base import (
    BasePhishingDetector,
    PredictionResult,
)


class URLGradientBoostingDetector(BasePhishingDetector):
    """
    Gradient Boosting detector for numerical URL features.

    URL preprocessing is handled separately by URLFeatureExtractor.
    The detector therefore receives an already transformed numerical
    feature matrix.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
        )

        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: Iterable[int],
    ) -> "URLGradientBoostingDetector":
        """
        Fit the Gradient Boosting model.
        """

        self.model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
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

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return class probabilities.
        """

        self._check_fitted()

        return self.model.predict_proba(X)

    def predict_one(
        self,
        X: np.ndarray,
    ) -> PredictionResult:
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
                "model_type": "gradient_boosting",
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample,
            },
        )

    def get_model_name(self) -> str:
        """
        Return the unique detector name.
        """

        return "url_gradient_boosting"

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
                "URLGradientBoostingDetector must be fitted "
                "before prediction."
            )