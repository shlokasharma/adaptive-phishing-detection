"""
Evaluation metrics for phishing detection models.

This module provides a common evaluation interface for binary phishing
classification.

Label convention:

    0 = legitimate
    1 = phishing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """
    Standardized binary classification metrics.

    Attributes
    ----------
    accuracy:
        Overall classification accuracy.

    precision:
        Precision for the phishing class.

    recall:
        Recall for the phishing class.

    f1:
        F1-score for the phishing class.

    roc_auc:
        Receiver Operating Characteristic Area Under the Curve.

    pr_auc:
        Precision-Recall Area Under the Curve.

    confusion_matrix:
        Binary confusion matrix with the ordering:

            [[TN, FP],
             [FN, TP]]
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        """
        Convert metrics to a serializable dictionary.
        """

        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion_matrix,
        }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> ClassificationMetrics:
    """
    Calculate standardized binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels.

    y_pred:
        Predicted labels.

    y_probability:
        Probability of the phishing class (class 1).

    Returns
    -------
    ClassificationMetrics
        Standardized metric collection.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probability = np.asarray(y_probability)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    if len(y_true) != len(y_probability):
        raise ValueError(
            "y_true and y_probability must contain the same number "
            "of samples."
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError(
            "y_true must contain only binary labels: 0 and 1."
        )

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError(
            "y_pred must contain only binary labels: 0 and 1."
        )

    if np.any(y_probability < 0.0) or np.any(y_probability > 1.0):
        raise ValueError(
            "y_probability values must be between 0 and 1."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        confusion_matrix=matrix.astype(int).tolist(),
    )