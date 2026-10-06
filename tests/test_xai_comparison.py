"""
Tests for Phase 4.6 SHAP/LIME comparison.
"""

from phishing_detection.explainability.comparison import (
    compare_explanations,
)
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


def make_explanation(
    name,
    factors,
    runtime=10.0,
):
    return UnifiedExplanation(
        summary=f"{name} explanation",
        explainer_name=name,
        method=name.lower(),
        framework=name,
        scope=ExplanationScope.LOCAL,
        target=ExplanationTarget(
            label=1,
            label_name="phishing",
            probability=0.90,
        ),
        factors=factors,
        model_name="test_model",
        execution_time_ms=runtime,
        metadata={},
    )


def make_factor(
    feature,
    importance,
    direction,
    rank,
):
    return ExplanationFactor(
        feature=feature,
        value=feature,
        importance=importance,
        direction=direction,
        rank=rank,
        contribution=importance,
        metadata={},
    )


def test_basic_comparison():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "url",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "domain",
                0.70,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
            make_factor(
                "https",
                0.30,
                ExplanationDirection.SUPPORTS_LEGITIMATE,
                3,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "url",
                0.80,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "domain",
                0.60,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
            make_factor(
                "redirect",
                0.40,
                ExplanationDirection.SUPPORTS_PHISHING,
                3,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
        top_k=3,
    )

    assert result.factor_count_a == 3
    assert result.factor_count_b == 3
    assert result.top_k_overlap_count == 2
    assert result.top_k_overlap_ratio > 0.0


def test_explanation_names():
    shap = make_explanation(
        "shap",
        [],
    )

    lime = make_explanation(
        "lime",
        [],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.explainer_a == "shap"
    assert result.explainer_b == "lime"


def test_empty_explanations():
    shap = make_explanation(
        "shap",
        [],
    )

    lime = make_explanation(
        "lime",
        [],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.factor_count_a == 0
    assert result.factor_count_b == 0
    assert result.top_k_overlap_count == 0
    assert result.top_k_overlap_ratio == 0.0
    assert result.directional_agreement_ratio == 0.0


def test_top_k_validation():
    shap = make_explanation(
        "shap",
        [],
    )

    lime = make_explanation(
        "lime",
        [],
    )

    try:
        compare_explanations(
            shap,
            lime,
            top_k=0,
        )
        assert False
    except ValueError:
        assert True


def test_none_first_explanation():
    lime = make_explanation(
        "lime",
        [],
    )

    try:
        compare_explanations(
            None,
            lime,
        )
        assert False
    except ValueError:
        assert True


def test_none_second_explanation():
    shap = make_explanation(
        "shap",
        [],
    )

    try:
        compare_explanations(
            shap,
            None,
        )
        assert False
    except ValueError:
        assert True


def test_directional_agreement():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "url",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "domain",
                0.80,
                ExplanationDirection.SUPPORTS_LEGITIMATE,
                2,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "url",
                0.70,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "domain",
                0.50,
                ExplanationDirection.SUPPORTS_LEGITIMATE,
                2,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
        top_k=2,
    )

    assert result.directional_agreement_ratio == 1.0


def test_directional_disagreement():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "url",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "url",
                0.70,
                ExplanationDirection.SUPPORTS_LEGITIMATE,
                1,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
        top_k=1,
    )

    assert result.directional_agreement_ratio == 0.0


def test_importance_correlation():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "a",
                0.2,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.4,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
            make_factor(
                "c",
                0.8,
                ExplanationDirection.SUPPORTS_PHISHING,
                3,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "a",
                0.1,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.2,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
            make_factor(
                "c",
                0.4,
                ExplanationDirection.SUPPORTS_PHISHING,
                3,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
        top_k=3,
    )

    assert result.importance_correlation is not None
    assert result.importance_correlation > 0.99


def test_importance_correlation_requires_two_factors():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "url",
                0.9,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "url",
                0.8,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.importance_correlation is None


def test_sparsity():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "a",
                0.5,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.0,
                ExplanationDirection.NEUTRAL,
                2,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "a",
                0.5,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.2,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.explanation_sparsity_a == 0.5
    assert result.explanation_sparsity_b == 1.0


def test_runtime_comparison():
    shap = make_explanation(
        "shap",
        [],
        runtime=100.0,
    )

    lime = make_explanation(
        "lime",
        [],
        runtime=200.0,
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.execution_time_ms_a == 100.0
    assert result.execution_time_ms_b == 200.0
    assert result.runtime_ratio == 2.0


def test_zero_runtime_ratio():
    shap = make_explanation(
        "shap",
        [],
        runtime=0.0,
    )

    lime = make_explanation(
        "lime",
        [],
        runtime=20.0,
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.runtime_ratio is None


def test_factor_count_difference():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "a",
                0.5,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "a",
                0.5,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.3,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
            make_factor(
                "c",
                0.2,
                ExplanationDirection.SUPPORTS_PHISHING,
                3,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.factor_count_difference == 2


def test_to_dict():
    shap = make_explanation(
        "shap",
        [],
    )

    lime = make_explanation(
        "lime",
        [],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["explainer_a"] == "shap"
    assert data["explainer_b"] == "lime"
    assert "top_k_overlap_ratio" in data
    assert "importance_correlation" in data
    assert "runtime_ratio_b_over_a" in data


def test_case_normalisation():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "URL",
                0.9,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                " url ",
                0.8,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
        top_k=1,
    )

    assert result.top_k_overlap_count == 1


def test_metadata():
    shap = make_explanation(
        "shap",
        [],
    )

    lime = make_explanation(
        "lime",
        [],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.metadata["phase"] == "4.6"
    assert (
        result.metadata["comparison_type"]
        == "shap_vs_lime"
    )


def test_common_factor_count():
    shap = make_explanation(
        "shap",
        [
            make_factor(
                "a",
                0.8,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.5,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
        ],
    )

    lime = make_explanation(
        "lime",
        [
            make_factor(
                "a",
                0.7,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "c",
                0.4,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
        ],
    )

    result = compare_explanations(
        shap,
        lime,
    )

    assert result.metadata["common_factor_count"] == 1