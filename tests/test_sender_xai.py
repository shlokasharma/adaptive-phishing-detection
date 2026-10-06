"""
Tests for Phase 4.5 Sender XAI.
"""

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationScope,
)
from phishing_detection.explainability.sender_xai import (
    SenderXAIExplainer,
)


def make_sender_result():
    evidence = [
        Evidence(
            evidence_type="sender_domain_mismatch",
            source="sender_analysis_agent",
            description="Sender domain does not match the supplied URL domain.",
            value=True,
            polarity="supports_phishing",
            confidence=0.95,
            reliability=0.90,
        ),
        Evidence(
            evidence_type="spf",
            source="sender_analysis_agent",
            description="SPF validation passed.",
            value="pass",
            polarity="supports_legitimate",
            confidence=0.90,
            reliability=0.95,
        ),
    ]

    return AgentResult.from_probability(
        agent_name="sender_analysis_agent",
        modality="sender",
        phishing_probability=0.70,
        evidence=evidence,
        execution_metadata=None,
    )


def make_context():
    return AgentContext(
        input_text="Test email",
        sender="sender@example.com",
        url="https://example.com",
        metadata={},
    )


def test_sender_explainer_name():
    explainer = SenderXAIExplainer()

    assert (
        explainer.explainer_name
        == "sender_rule_explainer"
    )


def test_sender_method():
    explainer = SenderXAIExplainer()

    assert explainer.method == "rule_based"


def test_sender_framework():
    explainer = SenderXAIExplainer()

    assert explainer.framework == "native"


def test_sender_can_explain_sender_result():
    explainer = SenderXAIExplainer()

    result = make_sender_result()
    context = make_context()

    assert explainer.can_explain(
        result,
        context,
    )


def test_sender_cannot_explain_email_result():
    explainer = SenderXAIExplainer()

    result = AgentResult.from_probability(
        agent_name="email_analysis_agent",
        modality="email",
        phishing_probability=0.70,
        evidence=[],
        execution_metadata=None,
    )

    context = make_context()

    assert not explainer.can_explain(
        result,
        context,
    )


def test_sender_invalid_max_display():
    try:
        SenderXAIExplainer(
            max_display=0
        )
        assert False
    except ValueError:
        assert True


def test_sender_explanation_scope():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    assert (
        explanation.scope
        == ExplanationScope.LOCAL
    )


def test_sender_explanation_target():
    explainer = SenderXAIExplainer()

    result = make_sender_result()

    explanation = explainer.explain(
        result,
        make_context(),
    )

    assert (
        explanation.target.label
        == result.label
    )

    assert (
        explanation.target.label_name
        == result.label_name
    )

    assert (
        explanation.target.probability
        == result.phishing_probability
    )


def test_sender_explanation_contains_factors():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    assert explanation.factor_count == 2


def test_sender_phishing_factor():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    phishing_factors = (
        explanation.phishing_factors
    )

    assert len(phishing_factors) == 1

    assert (
        phishing_factors[0].feature
        == "sender_domain_mismatch"
    )


def test_sender_legitimate_factor():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    legitimate_factors = (
        explanation.legitimate_factors
    )

    assert len(legitimate_factors) == 1

    assert (
        legitimate_factors[0].feature
        == "spf"
    )


def test_sender_factor_importance():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    factor = explanation.factors[0]

    assert factor.importance >= 0.0
    assert factor.importance <= 1.0


def test_sender_factor_ranking():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    ranks = [
        factor.rank
        for factor in explanation.factors
    ]

    assert ranks == [1, 2]


def test_sender_summary():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    assert "phishing" in (
        explanation.summary.lower()
    )

    assert "legitimate" in (
        explanation.summary.lower()
    )


def test_sender_metadata():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    assert (
        explanation.explainer_name
        == "sender_rule_explainer"
    )

    assert (
        explanation.method
        == "rule_based"
    )

    assert (
        explanation.framework
        == "native"
    )

    assert (
        explanation.model_name
        == "sender_rule_analysis"
    )


def test_sender_factor_direction():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    assert (
        explanation.factors[0].direction
        == ExplanationDirection.SUPPORTS_PHISHING
    )

    assert (
        explanation.factors[1].direction
        == ExplanationDirection.SUPPORTS_LEGITIMATE
    )


def test_sender_to_dict():
    explainer = SenderXAIExplainer()

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    data = explanation.to_dict()

    assert isinstance(
        data,
        dict,
    )

    assert (
        data["explainer_name"]
        == "sender_rule_explainer"
    )

    assert (
        data["method"]
        == "rule_based"
    )


def test_sender_empty_evidence():
    explainer = SenderXAIExplainer()

    result = AgentResult.from_probability(
        agent_name="sender_analysis_agent",
        modality="sender",
        phishing_probability=0.50,
        evidence=[],
        execution_metadata=None,
    )

    explanation = explainer.explain(
        result,
        make_context(),
    )

    assert explanation.factor_count == 0

    assert (
        "did not produce"
        in explanation.summary
    )


def test_sender_max_display():
    explainer = SenderXAIExplainer(
        max_display=1
    )

    explanation = explainer.explain(
        make_sender_result(),
        make_context(),
    )

    assert explanation.factor_count == 1


def test_sender_none_result():
    explainer = SenderXAIExplainer()

    assert not explainer.can_explain(
        None,
        make_context(),
    )


def test_sender_none_context():
    explainer = SenderXAIExplainer()

    assert not explainer.can_explain(
        make_sender_result(),
        None,
    )