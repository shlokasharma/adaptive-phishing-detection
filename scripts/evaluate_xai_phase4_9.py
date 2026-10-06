"""
Run Phase 4.9 XAI evaluation on representative explanations.

This script validates the quantitative XAI evaluation framework before
running larger experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

from phishing_detection.explainability.evaluation import (
    XAIQualityEvaluator,
    XAIQualityReport,
)
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "baselines"
    / "outputs"
    / "phase4_9_xai_evaluation.json"
)


def make_demo_explanation(
    method: str,
) -> UnifiedExplanation:

    factors = [
        ExplanationFactor(
            feature="urgency_language",
            value="urgent",
            importance=0.90,
            direction=(
                ExplanationDirection.SUPPORTS_PHISHING
            ),
            rank=1,
            contribution=0.90,
            metadata={},
        ),
        ExplanationFactor(
            feature="credential_request",
            value="verify account",
            importance=0.80,
            direction=(
                ExplanationDirection.SUPPORTS_PHISHING
            ),
            rank=2,
            contribution=0.80,
            metadata={},
        ),
        ExplanationFactor(
            feature="sender_domain",
            value="example.com",
            importance=0.50,
            direction=(
                ExplanationDirection.SUPPORTS_PHISHING
            ),
            rank=3,
            contribution=0.50,
            metadata={},
        ),
    ]

    return UnifiedExplanation(
        summary=(
            "Representative phishing explanation "
            f"generated for {method} evaluation."
        ),
        explainer_name=f"{method}_explainer",
        method=method,
        framework=method.upper(),
        scope=ExplanationScope.LOCAL,
        target=ExplanationTarget(
            label=1,
            label_name="phishing",
            probability=0.94,
        ),
        factors=factors,
        model_name="phase2_baseline",
        execution_time_ms=1.0,
        metadata={
            "phase": "4.9",
            "evaluation": True,
        },
    )


def main() -> None:

    evaluator = XAIQualityEvaluator(
        top_k=3
    )

    shap_explanation = make_demo_explanation(
        "shap"
    )

    lime_explanation = make_demo_explanation(
        "lime"
    )

    report = XAIQualityReport()

    report.sparsity.append(
        evaluator.evaluate_sparsity(
            shap_explanation
        )
    )

    report.agreement.append(
        evaluator.compare_explainers(
            shap_explanation,
            lime_explanation,
        )
    )

    report.consistency.append(
        evaluator.evaluate_consistency(
            [
                shap_explanation,
                make_demo_explanation("shap"),
                make_demo_explanation("shap"),
            ]
        )
    )

    report.fidelity.append(
        evaluator.evaluate_fidelity(
            shap_explanation,
            original_probability=0.94,
            perturbed_probability=0.30,
        )
    )

    report.runtime.append(
        evaluator.evaluate_runtime(
            lambda: make_demo_explanation(
                "shap"
            ),
            runs=5,
        )
    )

    report.robustness.append(
        evaluator.evaluate_robustness(
            explanation_fn=lambda: make_demo_explanation(
                "shap"
            ),
            prediction_fn=lambda: 0.94,
            runs=5,
        )
    )

    evaluator.summarize(report)

    report.metadata.update(
        {
            "phase": "4.9",
            "evaluation_type": "xai_quality_evaluation",
            "top_k": 3,
            "status": "completed",
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report.to_dict(),
            file,
            indent=2,
        )

    print("=" * 72)
    print("PHASE 4.9 - XAI EVALUATION")
    print("=" * 72)

    print(
        f"Mean fidelity: "
        f"{report.summary.get('mean_fidelity', 0.0):.4f}"
    )

    print(
        f"Mean consistency: "
        f"{report.summary.get('mean_consistency', 0.0):.4f}"
    )

    print(
        "Mean SHAP/LIME agreement: "
        f"{report.summary.get('mean_shap_lime_agreement', 0.0):.4f}"
    )

    print(
        f"Mean runtime (ms): "
        f"{report.summary.get('mean_runtime_ms', 0.0):.4f}"
    )

    print(
        f"Mean sparsity: "
        f"{report.summary.get('mean_sparsity', 0.0):.4f}"
    )

    print(
        f"Mean robustness: "
        f"{report.summary.get('mean_robustness', 0.0):.4f}"
    )

    print("-" * 72)
    print(f"Output: {OUTPUT_PATH}")
    print("STATUS: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()