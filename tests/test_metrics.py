"""
Unit tests for phishing detection evaluation metrics.
"""

import numpy as np
import pytest

from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)


def test_metrics_are_calculated_correctly():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 0, 1, 1]
    )

    y_probability = np.array(
        [0.05, 0.10, 0.90, 0.95]
    )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0

    assert metrics.confusion_matrix == [
        [2, 0],
        [0, 2],
    ]


def test_metrics_can_be_serialized():
    y_true = np.array(
        [0, 1, 0, 1]
    )

    y_pred = np.array(
        [0, 1, 1, 1]
    )

    y_probability = np.array(
        [0.10, 0.90, 0.70, 0.80]
    )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    result = metrics.to_dict()

    assert isinstance(result, dict)

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "roc_auc" in result
    assert "pr_auc" in result
    assert "confusion_matrix" in result


def test_mismatched_prediction_length_is_rejected():
    y_true = np.array(
        [0, 1, 0]
    )

    y_pred = np.array(
        [0, 1]
    )

    y_probability = np.array(
        [0.10, 0.90, 0.20]
    )

    with pytest.raises(ValueError):
        calculate_classification_metrics(
            y_true,
            y_pred,
            y_probability,
        )


def test_invalid_labels_are_rejected():
    y_true = np.array(
        [0, 1, 2]
    )

    y_pred = np.array(
        [0, 1, 1]
    )

    y_probability = np.array(
        [0.10, 0.90, 0.80]
    )

    with pytest.raises(ValueError):
        calculate_classification_metrics(
            y_true,
            y_pred,
            y_probability,
        )


def test_invalid_probabilities_are_rejected():
    y_true = np.array(
        [0, 1, 0]
    )

    y_pred = np.array(
        [0, 1, 0]
    )

    y_probability = np.array(
        [0.10, 1.20, 0.20]
    )

    with pytest.raises(ValueError):
        calculate_classification_metrics(
            y_true,
            y_pred,
            y_probability,
        )