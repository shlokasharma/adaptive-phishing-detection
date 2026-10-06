"""
Tests for the Phase 2 baseline experiment configuration.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase2_baseline.yaml"
)


def test_phase2_config_exists():
    assert CONFIG_FILE.exists()


def test_phase2_config_is_not_empty():
    assert CONFIG_FILE.stat().st_size > 0


def test_phase2_config_contains_core_sections():
    content = CONFIG_FILE.read_text(
        encoding="utf-8"
    )

    required_sections = [
        "experiment:",
        "label_convention:",
        "data:",
        "features:",
        "models:",
        "model_selection:",
        "evaluation:",
        "experiments:",
        "artifacts:",
        "reproducibility:",
    ]

    for section in required_sections:
        assert section in content


def test_phase2_config_records_selected_models():
    content = CONFIG_FILE.read_text(
        encoding="utf-8"
    )

    assert "email_linear_svm" in content
    assert "url_random_forest" in content


def test_phase2_config_freezes_test_set_usage():
    content = CONFIG_FILE.read_text(
        encoding="utf-8"
    )

    assert (
        "test_set_used_for_model_selection: false"
        in content
    )

    assert (
        "test_set_used_during_tuning: false"
        in content
    )