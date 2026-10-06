"""
TF-IDF + Random Forest phishing email detector.

This module implements the third Phase 2 traditional machine-learning
baseline for email phishing detection.

Pipeline:

    Raw email text
        ↓
    TF-IDF representation
        ↓
    Random Forest
        ↓
    Phishing probability
        ↓
    Standardized PredictionResult
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier

from phishing_detection.models.base import (
    BasePhishingDetector,
    PredictionResult,
)


class EmailRandomForestDetector(BasePhishingDetector):
    """
    Random Forest detector for phishing email classification.

    Input:
        TF-IDF feature matrix generated from email text.

    Output:
        Standardized PredictionResult.

    Label convention:
        0 = legitimate
        1 = phishing.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
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
        X: csr_matrix,
        y: Iterable[int],
    ) -> "EmailRandomForestDetector":
        """
        Train the Random Forest detector.
        """

        self.model.fit(X, y)

        self._is_fitted = True

        return self

    def predict(
        self,
        X: csr_matrix,
    ) -> np.ndarray:
        """
        Predict phishing/legitimate labels.
        """

        self._check_fitted()

        return self.model.predict(X)

    def predict_proba(
        self,
        X: csr_matrix,
    ) -> np.ndarray:
        """
        Return class probabilities.

        Column 0:
            probability of legitimate.

        Column 1:
            probability of phishing.
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
        Return the detector name.
        """

        return "email_tfidf_random_forest"

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
                "EmailRandomForestDetector must be fitted "
                "before prediction."
            )