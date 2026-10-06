"""
Tests for Phase 2 confidence-analysis utilities and artifacts.
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


def test_confidence_analysis_output_schema():
    path = (
        OUTPUT_DIR
        / "model_confidence_analysis.json"
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
        == "phase2_model_confidence_analysis"
    )

    assert (
        data["confidence_definition"]
        == "max(P(legitimate), P(phishing))"
    )

    assert (
        data["thresholds_are_descriptive"]
        is True
    )

    assert "email" in data["models"]
    assert "url" in data["models"]


def test_confidence_bands_are_valid():
    path = (
        OUTPUT_DIR
        / "model_confidence_analysis.json"
    )

    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    for modality in ["email", "url"]:

        bands = data["models"][modality][
            "confidence_bands"
        ]

        total_rate = sum(
            bands[name]["rate"]
            for name in [
                "low",
                "medium",
                "high",
            ]
        )

        assert abs(total_rate - 1.0) < 1e-6


def test_confidence_rates_are_bounded():
    path = (
        OUTPUT_DIR
        / "model_confidence_analysis.json"
    )

    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    for modality in ["email", "url"]:

        result = data["models"][modality]

        assert 0.0 <= result[
            "confidence_distribution"
        ]["mean"] <= 1.0

        assert 0.0 <= result[
            "confidence_distribution"
        ]["median"] <= 1.0

        bands = result["confidence_bands"]

        for band in ["low", "medium", "high"]:
            assert 0.0 <= bands[band]["rate"] <= 1.0