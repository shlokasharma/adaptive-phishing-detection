"""
Tests for Phase 2 validation evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
    / "validation_metrics.json"
)


EXPECTED_MODELS = {
    "email_logistic_regression",
    "email_linear_svm",
    "email_random_forest",
    "url_logistic_regression",
    "url_random_forest",
    "url_gradient_boosting",
}


REQUIRED_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "confusion_matrix",
}


def load_results() -> dict:
    assert RESULTS_FILE.exists(), (
        f"Validation results file not found: "
        f"{RESULTS_FILE}"
    )

    with RESULTS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_validation_results_file_exists():
    assert RESULTS_FILE.exists()


def test_all_six_models_are_present():

    payload = load_results()

    results = payload["results"]

    model_names = {
        result["model_name"]
        for result in results
    }

    assert model_names == EXPECTED_MODELS


def test_all_required_metrics_are_present():

    payload = load_results()

    for result in payload["results"]:

        missing = (
            REQUIRED_METRICS
            - set(result.keys())
        )

        assert not missing, (
            f"{result['model_name']} is missing "
            f"metrics: {missing}"
        )


def test_metric_values_are_valid():

    payload = load_results()

    probability_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }

    for result in payload["results"]:

        for metric in probability_metrics:

            value = result[metric]

            assert 0.0 <= value <= 1.0, (
                f"Invalid {metric} for "
                f"{result['model_name']}: {value}"
            )


def test_confusion_matrix_shape():

    payload = load_results()

    for result in payload["results"]:

        matrix = result["confusion_matrix"]

        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert len(matrix[1]) == 2