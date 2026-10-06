"""
Feature-based Logistic Regression phishing URL detector.

This module implements a Phase 2 traditional machine-learning
baseline for phishing URL detection.

Pipeline:

    Numerical URL features
        ↓
    Logistic Regression
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


class URLLogisticRegressionDetector(BasePhishingDetector):
    """
    Logistic Regression detector for numerical URL features.

    The feature preprocessing step is intentionally kept outside this
    detector. URLFeatureExtractor is responsible for fitting the
    preprocessing transformation on training data and transforming
    validation/test data.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="liblinear",
        )

        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: Iterable[int],
    ) -> "URLLogisticRegressionDetector":
        """
        Fit the Logistic Regression model.
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
                "model_type": "logistic_regression",
                "solver": "liblinear",
                "C": self.C,
                "max_iter": self.max_iter,
            },
        )

    def get_model_name(self) -> str:
        """
        Return the unique detector name.
        """

        return "url_logistic_regression"

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
                "URLLogisticRegressionDetector must be fitted "
                "before prediction."
            )