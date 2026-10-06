"""
TF-IDF + Linear SVM phishing email detector.

This module implements the second Phase 2 traditional machine-learning
baseline for email phishing detection.

Because LinearSVC does not natively expose calibrated probabilities,
CalibratedClassifierCV is used so that the detector can provide
probability estimates through the common phishing detector interface.

Pipeline:

    Raw email text
        ↓
    TF-IDF representation
        ↓
    Linear SVM
        ↓
    Probability calibration
        ↓
    Phishing probability
        ↓
    Standardized PredictionResult
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from phishing_detection.models.base import (
    BasePhishingDetector,
    PredictionResult,
)


class EmailLinearSVMDetector(BasePhishingDetector):
    """
    Calibrated Linear SVM detector for phishing email classification.

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
        max_iter: int = 2000,
        random_state: int = 42,
        calibration_cv: int = 3,
    ) -> None:
        self.base_model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )

        self.model = CalibratedClassifierCV(
            estimator=self.base_model,
            method="sigmoid",
            cv=calibration_cv,
        )

        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.calibration_cv = calibration_cv

        self._is_fitted = False

    def fit(
        self,
        X: csr_matrix,
        y: Iterable[int],
    ) -> "EmailLinearSVMDetector":
        """
        Train and calibrate the Linear SVM detector.
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
        Return calibrated class probabilities.

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
                "model_type": "linear_svm",
                "calibration_method": "sigmoid",
                "calibration_cv": self.calibration_cv,
                "C": self.C,
                "max_iter": self.max_iter,
            },
        )

    def get_model_name(self) -> str:
        """
        Return the detector name.
        """

        return "email_tfidf_linear_svm"

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
                "EmailLinearSVMDetector must be fitted "
                "before prediction."
            )