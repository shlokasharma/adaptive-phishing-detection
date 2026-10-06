"""
Tests for Phase 4.7 multi-agent explanation aggregation.
"""

from phishing_detection.explainability.aggregation import (
    MultiAgentExplanationAggregator,
)
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


def make_factor(
    feature,
    importance,
    direction,
    rank=1,
):
    return ExplanationFactor(
        feature=feature,
        value=feature,
        importance=importance,
        direction=direction,
        rank=rank,
        contribution=importance,
        metadata={
            "source": "test_agent"
        },
    )


def make_explanation(
    explainer_name,
    model_name,
    probability,
    factors,
    agent_name,
):
    return UnifiedExplanation(
        summary="Test explanation",
        explainer_name=explainer_name,
        method=explainer_name,
        framework=explainer_name,
        scope=ExplanationScope.LOCAL,
        target=ExplanationTarget(
            label=int(
                probability >= 0.5
            ),
            label_name=(
                "phishing"
                if probability >= 0.5
                else "legitimate"
            ),
            probability=probability,
        ),
        factors=factors,
        model_name=model_name,
        execution_time_ms=10.0,
        metadata={
            "agent_name": agent_name
        },
    )


def test_aggregator_name():
    aggregator = (
        MultiAgentExplanationAggregator()
    )

    assert (
        aggregator.explainer_name
        == "multi_agent_explanation_aggregator"
    )


def test_aggregator_method():
    aggregator = (
        MultiAgentExplanationAggregator()
    )

    assert (
        aggregator.method
        == "multi_agent_aggregation"
    )


def test_aggregator_framework():
    aggregator = (
        MultiAgentExplanationAggregator()
    )

    assert (
        aggregator.framework
        == "custom"
    )


def test_basic_aggregation():
    email = make_explanation(
        "email_shap_explainer",
        "email_linear_svm",
        0.90,
        [
            make_factor(
                "suspicious",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    url = make_explanation(
        "url_shap_explainer",
        "url_random_forest",
        0.80,
        [
            make_factor(
                "redirect",
                0.70,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "url_analysis_agent",
    )

    aggregator = (
        MultiAgentExplanationAggregator()
    )

    result = aggregator.aggregate(
        [email, url]
    )

    assert result.factor_count == 2


def test_multiple_agents_are_preserved():
    email = make_explanation(
        "email_shap_explainer",
        "email_model",
        0.80,
        [
            make_factor(
                "domain",
                0.80,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    url = make_explanation(
        "url_shap_explainer",
        "url_model",
        0.90,
        [
            make_factor(
                "url",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "url_analysis_agent",
    )

    sender = make_explanation(
        "sender_rule_explainer",
        "sender_rule_analysis",
        0.70,
        [
            make_factor(
                "spf",
                0.60,
                ExplanationDirection.SUPPORTS_LEGITIMATE,
            )
        ],
        "sender_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [
                email,
                url,
                sender,
            ]
        )
    )

    assert (
        result.metadata["agent_count"]
        == 3
    )


def test_duplicate_factor_is_aggregated():
    email = make_explanation(
        "email_shap_explainer",
        "email_model",
        0.90,
        [
            make_factor(
                "domain",
                0.50,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    url = make_explanation(
        "url_shap_explainer",
        "url_model",
        0.90,
        [
            make_factor(
                "domain",
                0.40,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "url_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email, url]
        )
    )

    assert result.factor_count == 1

    factor = result.factors[0]

    assert factor.feature == "domain"
    assert factor.importance == 0.90
    assert factor.metadata["support_count"] == 2


def test_agent_provenance_is_preserved():
    email = make_explanation(
        "email_shap_explainer",
        "email_model",
        0.90,
        [
            make_factor(
                "domain",
                0.50,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    url = make_explanation(
        "url_shap_explainer",
        "url_model",
        0.90,
        [
            make_factor(
                "domain",
                0.40,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "url_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email, url]
        )
    )

    factor = result.factors[0]

    assert (
        "email_analysis_agent"
        in factor.metadata["agents"]
    )

    assert (
        "url_analysis_agent"
        in factor.metadata["agents"]
    )


def test_explainer_provenance_is_preserved():
    email = make_explanation(
        "email_shap_explainer",
        "email_model",
        0.90,
        [
            make_factor(
                "domain",
                0.50,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    lime = make_explanation(
        "email_lime_explainer",
        "email_model",
        0.90,
        [
            make_factor(
                "domain",
                0.40,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email, lime]
        )
    )

    factor = result.factors[0]

    assert (
        "email_shap_explainer"
        in factor.metadata["explainers"]
    )

    assert (
        "email_lime_explainer"
        in factor.metadata["explainers"]
    )


def test_direction_is_preserved():
    phishing = make_explanation(
        "shap",
        "email_model",
        0.90,
        [
            make_factor(
                "url",
                0.80,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    legitimate = make_explanation(
        "sender",
        "sender_model",
        0.40,
        [
            make_factor(
                "spf",
                0.70,
                ExplanationDirection.SUPPORTS_LEGITIMATE,
            )
        ],
        "sender_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [
                phishing,
                legitimate,
            ]
        )
    )

    assert (
        result.factors[0].direction
        == ExplanationDirection.SUPPORTS_LEGITIMATE
        or result.factors[0].direction
        == ExplanationDirection.SUPPORTS_PHISHING
    )


def test_target_probability_is_aggregated():
    email = make_explanation(
        "email",
        "email_model",
        0.80,
        [],
        "email_analysis_agent",
    )

    url = make_explanation(
        "url",
        "url_model",
        0.60,
        [],
        "url_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email, url]
        )
    )

    assert (
        result.target.probability
        == 0.70
    )


def test_target_label_is_phishing():
    email = make_explanation(
        "email",
        "email_model",
        0.90,
        [],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email]
        )
    )

    assert result.target.label == 1
    assert (
        result.target.label_name
        == "phishing"
    )


def test_target_label_is_legitimate():
    email = make_explanation(
        "email",
        "email_model",
        0.20,
        [],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email]
        )
    )

    assert result.target.label == 0
    assert (
        result.target.label_name
        == "legitimate"
    )


def test_max_display():
    explanation = make_explanation(
        "email",
        "email_model",
        0.90,
        [
            make_factor(
                "a",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
                1,
            ),
            make_factor(
                "b",
                0.80,
                ExplanationDirection.SUPPORTS_PHISHING,
                2,
            ),
            make_factor(
                "c",
                0.70,
                ExplanationDirection.SUPPORTS_PHISHING,
                3,
            ),
        ],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator(
            max_display=2
        )
        .aggregate(
            [explanation]
        )
    )

    assert result.factor_count == 2


def test_empty_input_rejected():
    aggregator = (
        MultiAgentExplanationAggregator()
    )

    try:
        aggregator.aggregate([])
        assert False
    except ValueError:
        assert True


def test_none_input_rejected():
    aggregator = (
        MultiAgentExplanationAggregator()
    )

    try:
        aggregator.aggregate(None)
        assert False
    except ValueError:
        assert True


def test_invalid_object_rejected():
    aggregator = (
        MultiAgentExplanationAggregator()
    )

    try:
        aggregator.aggregate(
            ["not_an_explanation"]
        )
        assert False
    except ValueError:
        assert True


def test_summary_contains_agent_information():
    email = make_explanation(
        "email",
        "email_model",
        0.90,
        [
            make_factor(
                "url",
                0.80,
                ExplanationDirection.SUPPORTS_PHISHING,
            )
        ],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email]
        )
    )

    assert "agent(s)" in result.summary
    assert "explainer(s)" in result.summary


def test_metadata():
    email = make_explanation(
        "email",
        "email_model",
        0.90,
        [],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email]
        )
    )

    assert (
        result.metadata["phase"]
        == "4.7"
    )

    assert (
        result.metadata[
            "aggregation_type"
        ]
        == "multi_agent_evidence_aggregation"
    )


def test_to_dict():
    email = make_explanation(
        "email",
        "email_model",
        0.90,
        [],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [email]
        )
    )

    data = result.to_dict()

    assert isinstance(
        data,
        dict,
    )

    assert (
        data["explainer_name"]
        == "multi_agent_explanation_aggregator"
    )


def test_explain_alias():
    email = make_explanation(
        "email",
        "email_model",
        0.90,
        [],
        "email_analysis_agent",
    )

    aggregator = (
        MultiAgentExplanationAggregator()
    )

    result = aggregator.explain(
        [email]
    )

    assert isinstance(
        result,
        UnifiedExplanation,
    )


def test_neutral_factor_is_preserved():
    explanation = make_explanation(
        "email",
        "email_model",
        0.50,
        [
            make_factor(
                "unknown",
                0.30,
                ExplanationDirection.NEUTRAL,
            )
        ],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [explanation]
        )
    )

    assert (
        result.factors[0].direction
        == ExplanationDirection.NEUTRAL
    )


def test_factor_ranking():
    explanation = make_explanation(
        "email",
        "email_model",
        0.90,
        [
            make_factor(
                "weak",
                0.10,
                ExplanationDirection.SUPPORTS_PHISHING,
            ),
            make_factor(
                "strong",
                0.90,
                ExplanationDirection.SUPPORTS_PHISHING,
            ),
        ],
        "email_analysis_agent",
    )

    result = (
        MultiAgentExplanationAggregator()
        .aggregate(
            [explanation]
        )
    )

    assert (
        result.factors[0].feature
        == "strong"
    )

    assert result.factors[0].rank == 1