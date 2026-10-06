"""
TF-IDF + Logistic Regression phishing email detector.

This module implements the first Phase 2 traditional machine-learning
baseline for email phishing detection.

Pipeline:

    Raw email text
        ↓
    TF-IDF representation
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
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from phishing_detection.models.base import (
    BasePhishingDetector,
    PredictionResult,
)


class EmailLogisticRegressionDetector(BasePhishingDetector):
    """
    Logistic Regression detector for phishing email classification.

    Input:
        TF-IDF feature matrix generated from email text.

    Output:
        Standardized PredictionResult.

    Label convention:
        0 = legitimate
        1 = phishing
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="liblinear",
        )

        self._is_fitted = False

    def fit(
        self,
        X: csr_matrix,
        y: Iterable[int],
    ) -> "EmailLogisticRegressionDetector":
        """
        Train Logistic Regression on TF-IDF features.
        """

        self.model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, X: csr_matrix) -> np.ndarray:
        """
        Predict phishing/legitimate labels.
        """

        self._check_fitted()

        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """
        Return class probabilities.

        Column 0:
            probability of legitimate

        Column 1:
            probability of phishing
        """

        self._check_fitted()

        return self.model.predict_proba(X)

    def predict_one(
        self,
        X: csr_matrix,
    ) -> PredictionResult:
        """
        Generate a standardized prediction for one email.

        Parameters
        ----------
        X:
            TF-IDF representation of exactly one email.
        """

        self._check_fitted()

        probabilities = self.predict_proba(X)

        phishing_probability = float(probabilities[0, 1])

        label = int(self.predict(X)[0])

        return self.build_prediction_result(
            label=label,
            phishing_probability=phishing_probability,
            metadata={
                "model_type": "logistic_regression",
                "solver": "liblinear",
                "C": self.model.C,
                "max_iter": self.model.max_iter,
            },
        )

    def get_model_name(self) -> str:
        """
        Return the detector name.
        """

        return "email_tfidf_logistic_regression"

    def get_modality(self) -> str:
        """
        Return the detector modality.
        """

        return "email"

    def _check_fitted(self) -> None:
        """
        Ensure the detector has been trained before prediction.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "EmailLogisticRegressionDetector must be fitted "
                "before prediction."
            )