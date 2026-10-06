import pytest

from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


def make_target() -> ExplanationTarget:
    return ExplanationTarget(
        label=1,
        label_name="phishing",
        probability=0.92,
    )


def make_factor(
    *,
    direction: ExplanationDirection = (
        ExplanationDirection.SUPPORTS_PHISHING
    ),
) -> ExplanationFactor:
    return ExplanationFactor(
        feature="urgent",
        value="urgent",
        importance=0.31,
        direction=direction,
        rank=1,
        contribution=0.31,
    )


def make_explanation() -> UnifiedExplanation:
    return UnifiedExplanation(
        summary="The message is classified as phishing.",
        explainer_name="email_shap_explainer",
        method="shap",
        framework="SHAP",
        scope=ExplanationScope.LOCAL,
        target=make_target(),
        factors=[make_factor()],
        model_name="email_linear_svm",
        execution_time_ms=4.5,
    )


def test_explanation_direction_values():
    assert (
        ExplanationDirection.SUPPORTS_PHISHING.value
        == "supports_phishing"
    )
    assert (
        ExplanationDirection.SUPPORTS_LEGITIMATE.value
        == "supports_legitimate"
    )
    assert (
        ExplanationDirection.NEUTRAL.value
        == "neutral"
    )


def test_explanation_scope_values():
    assert ExplanationScope.LOCAL.value == "local"
    assert ExplanationScope.GLOBAL.value == "global"


def test_explanation_factor_creation():
    factor = make_factor()

    assert factor.feature == "urgent"
    assert factor.value == "urgent"
    assert factor.importance == 0.31
    assert (
        factor.direction
        == ExplanationDirection.SUPPORTS_PHISHING
    )
    assert factor.rank == 1
    assert factor.contribution == 0.31


def test_explanation_factor_to_dict():
    factor = make_factor()

    result = factor.to_dict()

    assert result["feature"] == "urgent"
    assert result["value"] == "urgent"
    assert result["importance"] == 0.31
    assert result["direction"] == "supports_phishing"
    assert result["rank"] == 1
    assert result["contribution"] == 0.31


def test_explanation_factor_accepts_string_direction():
    factor = ExplanationFactor(
        feature="url_length",
        value=120,
        importance=0.25,
        direction="supports_phishing",
        rank=1,
    )

    assert (
        factor.direction
        == ExplanationDirection.SUPPORTS_PHISHING
    )


def test_explanation_factor_rejects_empty_feature():
    with pytest.raises(ValueError):
        ExplanationFactor(
            feature="",
            value="test",
            importance=0.2,
            direction=ExplanationDirection.NEUTRAL,
            rank=1,
        )


def test_explanation_factor_rejects_invalid_rank():
    with pytest.raises(ValueError):
        ExplanationFactor(
            feature="test",
            value="test",
            importance=0.2,
            direction=ExplanationDirection.NEUTRAL,
            rank=0,
        )


def test_explanation_factor_rejects_invalid_direction():
    with pytest.raises(ValueError):
        ExplanationFactor(
            feature="test",
            value="test",
            importance=0.2,
            direction="invalid_direction",
            rank=1,
        )


def test_explanation_target_creation():
    target = make_target()

    assert target.label == 1
    assert target.label_name == "phishing"
    assert target.probability == 0.92


def test_explanation_target_to_dict():
    result = make_target().to_dict()

    assert result["label"] == 1
    assert result["label_name"] == "phishing"
    assert result["probability"] == 0.92


def test_explanation_target_rejects_invalid_label():
    with pytest.raises(ValueError):
        ExplanationTarget(
            label=2,
            label_name="invalid",
            probability=0.5,
        )


def test_explanation_target_rejects_invalid_probability():
    with pytest.raises(ValueError):
        ExplanationTarget(
            label=1,
            label_name="phishing",
            probability=1.2,
        )


def test_explanation_target_rejects_empty_label_name():
    with pytest.raises(ValueError):
        ExplanationTarget(
            label=1,
            label_name="",
            probability=0.5,
        )


def test_unified_explanation_creation():
    explanation = make_explanation()

    assert (
        explanation.explainer_name
        == "email_shap_explainer"
    )
    assert explanation.method == "shap"
    assert explanation.framework == "SHAP"
    assert explanation.scope == ExplanationScope.LOCAL
    assert explanation.model_name == "email_linear_svm"
    assert explanation.factor_count == 1


def test_unified_explanation_to_dict():
    result = make_explanation().to_dict()

    assert (
        result["explainer_name"]
        == "email_shap_explainer"
    )
    assert result["method"] == "shap"
    assert result["framework"] == "SHAP"
    assert result["scope"] == "local"
    assert result["target"]["label"] == 1
    assert len(result["factors"]) == 1


def test_unified_explanation_phishing_factors():
    explanation = make_explanation()

    assert len(explanation.phishing_factors) == 1
    assert len(explanation.legitimate_factors) == 0


def test_unified_explanation_legitimate_factors():
    explanation = UnifiedExplanation(
        summary="The message is legitimate.",
        explainer_name="email_lime_explainer",
        method="lime",
        framework="LIME",
        scope=ExplanationScope.LOCAL,
        target=ExplanationTarget(
            label=0,
            label_name="legitimate",
            probability=0.08,
        ),
        factors=[
            make_factor(
                direction=ExplanationDirection.SUPPORTS_LEGITIMATE
            )
        ],
    )

    assert len(explanation.phishing_factors) == 0
    assert len(explanation.legitimate_factors) == 1


def test_unified_explanation_accepts_string_scope():
    explanation = UnifiedExplanation(
        summary="Test explanation.",
        explainer_name="test_explainer",
        method="test",
        framework="Test",
        scope="local",
        target=make_target(),
        factors=[],
    )

    assert explanation.scope == ExplanationScope.LOCAL


def test_unified_explanation_rejects_empty_summary():
    with pytest.raises(ValueError):
        UnifiedExplanation(
            summary="",
            explainer_name="test",
            method="test",
            framework="test",
            scope=ExplanationScope.LOCAL,
            target=make_target(),
            factors=[],
        )


def test_unified_explanation_rejects_empty_explainer_name():
    with pytest.raises(ValueError):
        UnifiedExplanation(
            summary="Test.",
            explainer_name="",
            method="test",
            framework="test",
            scope=ExplanationScope.LOCAL,
            target=make_target(),
            factors=[],
        )


def test_unified_explanation_rejects_empty_method():
    with pytest.raises(ValueError):
        UnifiedExplanation(
            summary="Test.",
            explainer_name="test",
            method="",
            framework="test",
            scope=ExplanationScope.LOCAL,
            target=make_target(),
            factors=[],
        )


def test_unified_explanation_rejects_empty_framework():
    with pytest.raises(ValueError):
        UnifiedExplanation(
            summary="Test.",
            explainer_name="test",
            method="test",
            framework="",
            scope=ExplanationScope.LOCAL,
            target=make_target(),
            factors=[],
        )


def test_unified_explanation_rejects_negative_execution_time():
    with pytest.raises(ValueError):
        UnifiedExplanation(
            summary="Test.",
            explainer_name="test",
            method="test",
            framework="SHAP",
            scope=ExplanationScope.LOCAL,
            target=make_target(),
            factors=[],
            execution_time_ms=-1,
        )


def test_unified_explanation_global_scope():
    explanation = UnifiedExplanation(
        summary="Global feature importance.",
        explainer_name="url_shap_explainer",
        method="shap",
        framework="SHAP",
        scope=ExplanationScope.GLOBAL,
        target=make_target(),
        factors=[],
    )

    assert explanation.scope == ExplanationScope.GLOBAL