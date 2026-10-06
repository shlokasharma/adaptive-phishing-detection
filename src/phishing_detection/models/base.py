"""
Common interfaces for phishing detection models.

All Phase 2 baseline detectors should follow the interface defined here.
The interface is intentionally lightweight so that later components such
as specialized agents and adaptive orchestration can consume detector
outputs consistently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionResult:
    """
    Standardized prediction returned by every detector.

    Attributes
    ----------
    label:
        Predicted class.
        0 = legitimate
        1 = phishing

    label_name:
        Human-readable predicted class.

    confidence:
        Confidence associated with the predicted class.

    phishing_probability:
        Estimated probability that the sample is phishing.
        May be None for models that do not provide probabilities.

    model_name:
        Name of the detector that produced the prediction.

    modality:
        Evidence modality used by the detector, such as "email" or "url".

    metadata:
        Additional model-specific information.
    """

    label: int
    label_name: str
    confidence: float
    phishing_probability: float | None
    model_name: str
    modality: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePhishingDetector(ABC):
    """
    Abstract interface for all phishing detection models.

    Every detector should provide the following operations:

    - fit()
    - predict()
    - predict_proba()
    - get_model_name()
    - get_modality()
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BasePhishingDetector":
        """
        Train the detector.

        Parameters
        ----------
        X:
            Training features.

        y:
            Training labels.

        Returns
        -------
        BasePhishingDetector
            The fitted detector.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict class labels.

        Parameters
        ----------
        X:
            Input features.

        Returns
        -------
        Any
            Predicted labels.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """
        Predict class probabilities when supported.

        Parameters
        ----------
        X:
            Input features.

        Returns
        -------
        Any
            Probability estimates.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the unique detector name."""
        raise NotImplementedError

    @abstractmethod
    def get_modality(self) -> str:
        """Return the evidence modality used by the detector."""
        raise NotImplementedError

    def build_prediction_result(
        self,
        label: int,
        phishing_probability: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """
        Convert a model prediction into the common PredictionResult format.
        """

        label = int(label)

        if label not in (0, 1):
            raise ValueError(
                f"Invalid phishing label: {label}. "
                "Expected 0 (legitimate) or 1 (phishing)."
            )

        if phishing_probability is not None:
            phishing_probability = float(phishing_probability)

            if not 0.0 <= phishing_probability <= 1.0:
                raise ValueError(
                    "phishing_probability must be between 0 and 1."
                )

            confidence = (
                phishing_probability
                if label == 1
                else 1.0 - phishing_probability
            )
        else:
            confidence = 1.0

        label_name = (
            "phishing"
            if label == 1
            else "legitimate"
        )

        return PredictionResult(
            label=label,
            label_name=label_name,
            confidence=float(confidence),
            phishing_probability=phishing_probability,
            model_name=self.get_model_name(),
            modality=self.get_modality(),
            metadata=metadata or {},
        )