"""
Tests for Phase 2 classification-report utilities.
"""

import numpy as np

from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)


def test_classification_metrics_contains_required_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_probability = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    result = metrics.to_dict()

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "roc_auc" in result
    assert "pr_auc" in result
    assert "confusion_matrix" in result


def test_confusion_matrix_has_binary_structure():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_probability = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    matrix = metrics.confusion_matrix

    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert len(matrix[1]) == 2


def test_confusion_matrix_counts_are_correct():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_probability = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    # [[TN, FP],
    #  [FN, TP]]
    assert metrics.confusion_matrix == [
        [1, 1],
        [0, 2],
    ]


def test_perfect_predictions_have_perfect_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_probability = np.array([0.01, 0.10, 0.90, 0.99])

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0