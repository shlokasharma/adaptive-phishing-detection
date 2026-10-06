import numpy as np
import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.email_xai import (
    EmailLIMEExplainer,
    EmailSHAPExplainer,
    EmailXAIExplainer,
    _direction_from_contribution,
)
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationScope,
    UnifiedExplanation,
)


class FakeVectorizer:
    """Minimal vectorizer for unit tests."""

    def transform(self, texts):
        return np.asarray(
            [
                [len(text)]
                for text in texts
            ],
            dtype=float,
        )


class FakeModel:
    """Minimal binary classifier for unit tests."""

    def predict_proba(self, features):
        probabilities = []

        for row in features:
            value = float(row[0])

            phishing_probability = min(
                0.99,
                max(
                    0.01,
                    0.5 + (value / 1000.0),
                ),
            )

            probabilities.append(
                [
                    1.0 - phishing_probability,
                    phishing_probability,
                ]
            )

        return np.asarray(probabilities)


class FakeSHAPExplanation:
    """Fake SHAP output."""

    data = np.asarray(
        ["urgent", "verify", "account"]
    )

    values = np.asarray(
        [0.30, 0.20, -0.10]
    )


class FakeSHAPExplainer:
    """Fake SHAP explainer."""

    def __call__(self, texts, max_evals):
        assert len(texts) == 1
        assert max_evals > 0

        return [FakeSHAPExplanation()]


class FakeLIMEExplanation:
    """Fake LIME explanation."""

    def as_list(self, label):
        assert label == 1

        return [
            ("urgent", 0.30),
            ("verify", 0.20),
            ("account", -0.10),
        ]


class FakeLIMEExplainer:
    """Fake LIME explainer."""

    def explain_instance(
        self,
        text,
        predict_fn,
        num_features,
        num_samples,
        labels,
    ):
        assert text
        assert num_features > 0
        assert num_samples > 0
        assert labels == (1,)

        prediction = predict_fn([text])

        assert prediction.shape == (1, 2)

        return FakeLIMEExplanation()


def make_result() -> AgentResult:
    return AgentResult.from_probability(
        agent_name="email_analysis_agent",
        modality="email",
        phishing_probability=0.92,
    )


def make_context() -> AgentContext:
    return AgentContext(
        input_text=(
            "Urgent: verify your account immediately "
            "or your account will be suspended."
        )
    )


def test_positive_contribution_supports_phishing():
    assert (
        _direction_from_contribution(0.5)
        == ExplanationDirection.SUPPORTS_PHISHING
    )


def test_negative_contribution_supports_legitimate():
    assert (
        _direction_from_contribution(-0.5)
        == ExplanationDirection.SUPPORTS_LEGITIMATE
    )


def test_zero_contribution_is_neutral():
    assert (
        _direction_from_contribution(0.0)
        == ExplanationDirection.NEUTRAL
    )


def test_shap_explainer_name():
    explainer = EmailSHAPExplainer()

    assert (
        explainer.get_explainer_name()
        == "email_shap_explainer"
    )


def test_shap_method():
    explainer = EmailSHAPExplainer()

    assert explainer.get_method_name() == "shap"


def test_shap_framework():
    explainer = EmailSHAPExplainer()

    assert explainer.get_framework_name() == "SHAP"


def test_shap_rejects_invalid_max_evals():
    with pytest.raises(ValueError):
        EmailSHAPExplainer(max_evals=0)


def test_shap_rejects_invalid_max_display():
    with pytest.raises(ValueError):
        EmailSHAPExplainer(max_display=0)


def test_shap_requires_input_text():
    explainer = EmailSHAPExplainer()

    context = AgentContext(input_text="")
    result = make_result()

    with pytest.raises(ValueError):
        explainer.explain(result, context)


def test_shap_explanation_with_mocked_backend(monkeypatch):
    explainer = EmailSHAPExplainer()

    explainer._model = FakeModel()
    explainer._vectorizer = FakeVectorizer()
    explainer._explainer = FakeSHAPExplainer()

    explanation = explainer.explain(
        make_result(),
        make_context(),
    )

    assert isinstance(
        explanation,
        UnifiedExplanation,
    )

    assert explanation.method == "shap"
    assert explanation.framework == "SHAP"
    assert explanation.scope == ExplanationScope.LOCAL
    assert explanation.model_name == "email_linear_svm"

    assert explanation.factor_count == 3

    assert (
        explanation.factors[0].feature
        == "urgent"
    )

    assert (
        explanation.factors[0].direction
        == ExplanationDirection.SUPPORTS_PHISHING
    )

    assert (
        explanation.factors[2].direction
        == ExplanationDirection.SUPPORTS_LEGITIMATE
    )


def test_lime_explainer_name():
    explainer = EmailLIMEExplainer()

    assert (
        explainer.get_explainer_name()
        == "email_lime_explainer"
    )


def test_lime_method():
    explainer = EmailLIMEExplainer()

    assert explainer.get_method_name() == "lime"


def test_lime_framework():
    explainer = EmailLIMEExplainer()

    assert explainer.get_framework_name() == "LIME"


def test_lime_rejects_invalid_num_features():
    with pytest.raises(ValueError):
        EmailLIMEExplainer(num_features=0)


def test_lime_rejects_invalid_num_samples():
    with pytest.raises(ValueError):
        EmailLIMEExplainer(num_samples=0)


def test_lime_rejects_invalid_class_names():
    with pytest.raises(ValueError):
        EmailLIMEExplainer(
            class_names=["only_one_class"]
        )


def test_lime_requires_input_text():
    explainer = EmailLIMEExplainer()

    context = AgentContext(input_text="")
    result = make_result()

    with pytest.raises(ValueError):
        explainer.explain(result, context)


def test_lime_explanation_with_mocked_backend():
    explainer = EmailLIMEExplainer()

    explainer._model = FakeModel()
    explainer._vectorizer = FakeVectorizer()
    explainer._explainer = FakeLIMEExplainer()

    explanation = explainer.explain(
        make_result(),
        make_context(),
    )

    assert isinstance(
        explanation,
        UnifiedExplanation,
    )

    assert explanation.method == "lime"
    assert explanation.framework == "LIME"
    assert explanation.scope == ExplanationScope.LOCAL
    assert explanation.model_name == "email_linear_svm"

    assert explanation.factor_count == 3

    assert (
        explanation.factors[0].feature
        == "urgent"
    )

    assert (
        explanation.factors[0].direction
        == ExplanationDirection.SUPPORTS_PHISHING
    )


def test_email_xai_wrapper():
    shap_explainer = EmailSHAPExplainer(
        max_evals=10,
    )

    lime_explainer = EmailLIMEExplainer(
        num_features=3,
        num_samples=10,
    )

    shap_explainer._model = FakeModel()
    shap_explainer._vectorizer = FakeVectorizer()
    shap_explainer._explainer = FakeSHAPExplainer()

    lime_explainer._model = FakeModel()
    lime_explainer._vectorizer = FakeVectorizer()
    lime_explainer._explainer = FakeLIMEExplainer()

    wrapper = EmailXAIExplainer(
        shap_explainer=shap_explainer,
        lime_explainer=lime_explainer,
    )

    explanations = wrapper.explain(
        make_result(),
        make_context(),
    )

    assert set(explanations.keys()) == {
        "shap",
        "lime",
    }

    assert (
        explanations["shap"].framework
        == "SHAP"
    )

    assert (
        explanations["lime"].framework
        == "LIME"
    )


def test_shap_model_probability_function():
    explainer = EmailSHAPExplainer()

    explainer._model = FakeModel()
    explainer._vectorizer = FakeVectorizer()

    probabilities = explainer._predict_proba(
        ["short text"]
    )

    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0


def test_lime_model_probability_function():
    explainer = EmailLIMEExplainer()

    explainer._model = FakeModel()
    explainer._vectorizer = FakeVectorizer()

    probabilities = explainer._predict_proba(
        ["short text"]
    )

    assert probabilities.shape == (1, 2)
    assert np.isclose(
        probabilities[0].sum(),
        1.0,
    )