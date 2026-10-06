"""
Phase 2 — Reproducible baseline experiment runner.

This script provides a single entry point for the finalized
Phase 2 baseline analysis workflow.

The workflow operates on the frozen Phase 1 datasets and the
already-trained baseline model artifacts.

No hyperparameter tuning is performed by this runner.

No model retraining is performed by this runner.

Pipeline:

    1. Validation evaluation
    2. Held-out test evaluation
    3. Classification reports
    4. Error analysis
    5. Confidence analysis
    6. Computational-cost analysis
    7. Baseline comparison

Usage:

    python scripts/run_phase2_baseline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)


PIPELINE_STEPS = [
    (
        "Validation evaluation",
        "evaluate_baselines_validation.py",
    ),
    (
        "Held-out test evaluation",
        "evaluate_selected_models_test.py",
    ),
    (
        "Classification reports",
        "generate_test_classification_reports.py",
    ),
    (
        "Error analysis",
        "generate_error_analysis_summary.py",
    ),
    (
        "Confidence analysis",
        "analyze_model_confidence.py",
    ),
    (
        "Computational-cost analysis",
        "analyze_computational_cost.py",
    ),
    (
        "Baseline comparison",
        "generate_baseline_comparison.py",
    ),
]


def run_script(
    description: str,
    script_name: str,
) -> None:
    """
    Execute one Phase 2 analysis script.

    The subprocess inherits the current Python environment,
    ensuring that the active .venv interpreter is used.
    """

    script_path = (
        SCRIPTS_DIR
        / script_name
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required Phase 2 script not found:\n"
            f"{script_path}"
        )

    print("\n" + "=" * 80)
    print(
        f"PHASE 2 STEP — {description.upper()}"
    )
    print("=" * 80)

    print(
        f"Running: {script_path.name}"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Phase 2 pipeline step failed: "
            f"{description}\n"
            f"Script: {script_path.name}\n"
            f"Exit code: {result.returncode}"
        )

    print(
        f"Completed successfully: "
        f"{description}"
    )


def verify_required_artifacts() -> None:
    """
    Verify that the finalized baseline model artifacts exist
    before running the analysis pipeline.
    """

    model_directory = (
        PROJECT_ROOT
        / "experiments"
        / "baselines"
        / "models"
    )

    required_models = [
        "email_linear_svm.pkl",
        "url_random_forest.pkl",
    ]

    missing_models = [
        model
        for model in required_models
        if not (
            model_directory
            / model
        ).exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "Required selected baseline model artifacts "
            "are missing:\n"
            + "\n".join(
                f"  - {model}"
                for model in missing_models
            )
        )

    print(
        "Selected baseline model artifacts verified:"
    )

    for model in required_models:
        print(
            f"  ✓ {model}"
        )


def verify_phase2_outputs() -> None:
    """
    Verify that the major Phase 2 outputs were generated.
    """

    output_directory = (
        PROJECT_ROOT
        / "experiments"
        / "baselines"
        / "outputs"
    )

    required_outputs = [
        "validation_metrics.json",
        "test_metrics.json",
        "error_analysis_summary.csv",
        "model_confidence_analysis.json",
        "computational_cost_analysis.json",
        "baseline_model_comparison.csv",
        "baseline_model_comparison.json",
        "baseline_model_comparison.md",
    ]

    missing_outputs = [
        output
        for output in required_outputs
        if not (
            output_directory
            / output
        ).exists()
    ]

    if missing_outputs:
        raise RuntimeError(
            "Phase 2 pipeline completed, but the following "
            "expected outputs are missing:\n"
            + "\n".join(
                f"  - {output}"
                for output in missing_outputs
            )
        )

    print(
        "\nVerified Phase 2 output artifacts:"
    )

    for output in required_outputs:
        print(
            f"  ✓ {output}"
        )


def main() -> None:
    """Run the complete Phase 2 baseline analysis."""

    print("=" * 80)
    print(
        "PHASE 2 — REPRODUCIBLE BASELINE EXPERIMENT"
    )
    print("=" * 80)

    print(
        "\nExperiment properties:"
    )

    print(
        "  Phase 1 benchmark frozen : YES"
    )

    print(
        "  Hyperparameter tuning    : NO"
    )

    print(
        "  Model retraining         : NO"
    )

    print(
        "  Test-set model selection : NO"
    )

    print(
        "  Random state              : 42"
    )

    verify_required_artifacts()

    for description, script_name in PIPELINE_STEPS:
        run_script(
            description,
            script_name,
        )

    verify_phase2_outputs()

    print("\n" + "=" * 80)
    print(
        "PHASE 2 BASELINE EXPERIMENT COMPLETE"
    )
    print("=" * 80)

    print(
        "\nAll reproducible Phase 2 analysis stages "
        "completed successfully."
    )

    print(
        "\nPrimary comparison table:"
    )

    print(
        PROJECT_ROOT
        / "experiments"
        / "baselines"
        / "outputs"
        / "baseline_model_comparison.md"
    )


if __name__ == "__main__":
    main()