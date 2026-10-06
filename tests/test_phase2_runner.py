"""
Tests for the Phase 2 reproducible experiment runner.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)


def test_phase2_runner_exists():
    runner = (
        SCRIPTS_DIR
        / "run_phase2_baseline.py"
    )

    assert runner.exists()


def test_phase2_runner_contains_required_pipeline_steps():
    runner = (
        SCRIPTS_DIR
        / "run_phase2_baseline.py"
    )

    content = runner.read_text(
        encoding="utf-8"
    )

    required_scripts = [
        "evaluate_baselines_validation.py",
        "evaluate_selected_models_test.py",
        "generate_test_classification_reports.py",
        "generate_error_analysis_summary.py",
        "analyze_model_confidence.py",
        "analyze_computational_cost.py",
        "generate_baseline_comparison.py",
    ]

    for script_name in required_scripts:
        assert script_name in content


def test_phase2_runner_does_not_run_tuning():
    runner = (
        SCRIPTS_DIR
        / "run_phase2_baseline.py"
    )

    content = runner.read_text(
        encoding="utf-8"
    )

    assert "tune_baseline_models.py" not in content
    assert "tune_baseline_models_fast.py" not in content


def test_phase2_runner_requires_selected_models():
    runner = (
        SCRIPTS_DIR
        / "run_phase2_baseline.py"
    )

    content = runner.read_text(
        encoding="utf-8"
    )

    assert "email_linear_svm.pkl" in content
    assert "url_random_forest.pkl" in content