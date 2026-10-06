"""
Tests for Phase 2 cross-dataset evaluation artifacts.
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


def test_email_cross_dataset_output_schema():
    path = OUTPUT_DIR / "email_cross_dataset_metrics.json"

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["experiment"] == (
        "phase2_email_cross_dataset_generalization"
    )

    assert data["external_retraining"] is False
    assert data["external_hyperparameter_tuning"] is False

    assert "datasets" in data


def test_url_cross_dataset_compatibility_output():
    path = (
        OUTPUT_DIR
        / "url_cross_dataset_compatibility.json"
    )

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert "uci_feature_count" in data
    assert "phiusiil_feature_count" in data
    assert "schemas_directly_compatible" in data
    assert "direct_cross_dataset_evaluation_valid" in data


def test_cross_dataset_outputs_are_json_serializable():
    for filename in [
        "email_cross_dataset_metrics.json",
        "url_cross_dataset_compatibility.json",
    ]:
        path = OUTPUT_DIR / filename

        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        assert isinstance(data, dict)