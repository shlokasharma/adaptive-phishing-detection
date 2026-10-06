"""
Tests for Phase 4.9 XAI evaluation.
"""

from phishing_detection.explainability.evaluation import (
    XAIQualityEvaluator,
)
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


def make_explanation(
    method="shap",
    offset=0.0,
):
    factors = [
        ExplanationFactor(
            feature="feature_a",
            value="A",
            importance=0.90 + offset,
            direction=(
                ExplanationDirection.SUPPORTS_PHISHING
            ),
            rank=1,
            contribution=0.90 + offset,
            metadata={},
        ),
        ExplanationFactor(
            feature="feature_b",
            value="B",
            importance=0.70 + offset,
            direction=(
                ExplanationDirection.SUPPORTS_PHISHING
            ),
            rank=2,
            contribution=0.70 + offset,
            metadata={},
        ),
        ExplanationFactor(
            feature="feature_c",
            value="C",
            importance=0.20 + offset,
            direction=(
                ExplanationDirection.SUPPORTS_LEGITIMATE
            ),
            rank=3,
            contribution=-0.20,
            metadata={},
        ),
    ]

    return UnifiedExplanation(
        summary="Test explanation.",
        explainer_name=f"{method}_explainer",
        method=method,
        framework=method.upper(),
        scope=ExplanationScope.LOCAL,
        target=ExplanationTarget(
            label=1,
            label_name="phishing",
            probability=0.90,
        ),
        factors=factors,
        model_name="test_model",
        execution_time_ms=1.0,
        metadata={},
    )


def test_sparsity():
    evaluator = XAIQualityEvaluator(
        top_k=3
    )

    explanation = make_explanation()

    result = evaluator.evaluate_sparsity(
        explanation
    )

    assert result.factor_count == 3
    assert result.non_zero_factor_count == 3
    assert result.sparsity == 0.0


def test_shap_lime_agreement():
    evaluator = XAIQualityEvaluator(
        top_k=3
    )

    shap = make_explanation(
        method="shap"
    )

    lime = make_explanation(
        method="lime"
    )

    result = evaluator.compare_explainers(
        shap,
        lime,
    )

    assert result.factor_overlap == 1.0
    assert result.directional_agreement == 1.0
    assert result.importance_correlation == 1.0
    assert result.agreement_score == 1.0


def test_consistency():
    evaluator = XAIQualityEvaluator(
        top_k=3
    )

    explanations = [
        make_explanation(
            offset=0.0
        ),
        make_explanation(
            offset=0.01
        ),
        make_explanation(
            offset=-0.01
        ),
    ]

    result = evaluator.evaluate_consistency(
        explanations
    )

    assert result.runs == 3
    assert result.factor_overlap == 1.0
    assert result.directional_agreement == 1.0
    assert result.consistency_score > 0.99


def test_runtime():
    evaluator = XAIQualityEvaluator()

    result = evaluator.evaluate_runtime(
        lambda: make_explanation(),
        runs=3,
    )

    assert result.runs == 3
    assert result.mean_ms >= 0.0
    assert result.min_ms >= 0.0
    assert result.max_ms >= result.min_ms


def test_fidelity():
    evaluator = XAIQualityEvaluator(
        top_k=2
    )

    explanation = make_explanation()

    result = evaluator.evaluate_fidelity(
        explanation=explanation,
        original_probability=0.95,
        perturbed_probability=0.25,
    )

    assert result.original_probability == 0.95
    assert result.perturbed_probability == 0.25
    assert result.probability_change == 0.70
    assert result.fidelity_score > 0.0
    assert result.top_k == 2


def test_robustness():
    evaluator = XAIQualityEvaluator(
        top_k=3
    )

    result = evaluator.evaluate_robustness(
        explanation_fn=lambda: make_explanation(),
        prediction_fn=lambda: 0.90,
        runs=5,
    )

    assert result.runs == 5
    assert result.successful_runs == 5
    assert result.failure_count == 0
    assert result.factor_overlap == 1.0
    assert result.probability_std == 0.0
    assert result.robustness_score == 1.0


def test_complete_report_summary():
    evaluator = XAIQualityEvaluator()

    explanation = make_explanation()

    sparsity = evaluator.evaluate_sparsity(
        explanation
    )

    fidelity = evaluator.evaluate_fidelity(
        explanation,
        original_probability=0.9,
        perturbed_probability=0.3,
    )

    agreement = evaluator.compare_explainers(
        explanation,
        make_explanation(
            method="lime"
        ),
    )

    from phishing_detection.explainability.evaluation import (
        XAIQualityReport,
    )

    report = XAIQualityReport(
        fidelity=[fidelity],
        agreement=[agreement],
        sparsity=[sparsity],
    )

    summary = evaluator.summarize(
        report
    )

    assert "mean_fidelity" in summary
    assert "mean_shap_lime_agreement" in summary
    assert "mean_sparsity" in summary


def test_serialization():
    evaluator = XAIQualityEvaluator()

    explanation = make_explanation()

    sparsity = evaluator.evaluate_sparsity(
        explanation
    )

    assert isinstance(
        sparsity.to_dict(),
        dict,
    )

    assert "sparsity" in sparsity.to_dict()