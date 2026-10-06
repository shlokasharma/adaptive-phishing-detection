"""
Tests for Phase 2 computational-cost analysis artifacts.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)


def test_computational_cost_output_schema():
    path = (
        OUTPUT_DIR
        / "computational_cost_analysis.json"
    )

    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert (
        data["experiment"]
        == "phase2_computational_cost_analysis"
    )

    assert (
        data["retraining_performed"]
        is False
    )

    assert "models" in data
    assert len(data["models"]) == 2


def test_computational_cost_values_are_valid():
    path = (
        OUTPUT_DIR
        / "computational_cost_analysis.json"
    )

    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    for result in data["models"]:

        assert result["num_samples"] > 0
        assert result["num_features"] > 0

        assert (
            result[
                "feature_transformation_time_seconds"
            ] >= 0
        )

        assert (
            result[
                "prediction_time_seconds"
            ] >= 0
        )

        assert (
            result[
                "probability_time_seconds"
            ] >= 0
        )

        assert (
            result[
                "total_inference_time_seconds"
            ] >= 0
        )

        assert (
            result[
                "total_time_per_sample_ms"
            ] >= 0
        )

        assert (
            result[
                "throughput_samples_per_second"
            ] >= 0
        )


def test_selected_models_are_present():
    path = (
        OUTPUT_DIR
        / "computational_cost_analysis.json"
    )

    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    models = {
        result["model"]
        for result in data["models"]
    }

    assert "email_linear_svm" in models
    assert "url_random_forest" in models