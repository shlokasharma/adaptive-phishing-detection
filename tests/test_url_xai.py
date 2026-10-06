from __future__ import annotations

import numpy as np
import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationScope,
)
from phishing_detection.explainability.url_xai import (
    URLLIMEExplainer,
    URLSHAPExplainer,
    URLXAIExplainer,
    _direction_from_contribution,
    _model_probability,
)


class FakeModel:
    def predict_proba(self, values):
        values = np.asarray(values)

        return np.tile(
            np.array([[0.25, 0.75]]),
            (values.shape[0], 1),
        )


class FakeExtractor:
    def transform(self, values):
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()

        return np.asarray(values, dtype=float)


class FakeSHAPExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, values):
        values = np.asarray(values)

        return [
            np.full(values.shape, -0.2),
            np.full(values.shape, 0.2),
        ]


class FakeSHAPModule:
    class TreeExplainer:
        def __new__(cls, model):
            return FakeSHAPExplainer(model)


def make_context():
    return AgentContext(
        input_text="https://example.com",
        url="https://example.com",
        metadata={
            "url_features": {
                "url_length": 40,
                "having_ip_address": 0,
                "shortining_service": 0,
            }
        },
    )


def make_result():
    return AgentResult.from_probability(
        agent_name="url_analysis_agent",
        modality="url",
        phishing_probability=0.75,
        evidence=[],
        execution_metadata=None,
    )


def test_direction_positive():
    assert (
        _direction_from_contribution(0.5)
        == ExplanationDirection.SUPPORTS_PHISHING
    )


def test_direction_negative():
    assert (
        _direction_from_contribution(-0.5)
        == ExplanationDirection.SUPPORTS_LEGITIMATE
    )


def test_direction_zero():
    assert (
        _direction_from_contribution(0.0)
        == ExplanationDirection.NEUTRAL
    )


def test_model_probability():
    probability = _model_probability(
        FakeModel(),
        np.array([[1.0, 2.0, 3.0]]),
    )

    assert probability == pytest.approx(0.75)


def test_shap_invalid_max_display():
    with pytest.raises(ValueError):
        URLSHAPExplainer(max_display=0)


def test_lime_invalid_num_features():
    with pytest.raises(ValueError):
        URLLIMEExplainer(num_features=0)


def test_lime_invalid_num_samples():
    with pytest.raises(ValueError):
        URLLIMEExplainer(num_samples=0)


def test_lime_invalid_background_size():
    with pytest.raises(ValueError):
        URLLIMEExplainer(background_size=0)


def test_shap_can_explain():
    explainer = URLSHAPExplainer()

    assert explainer.can_explain(
        make_result(),
        make_context(),
    )


def test_shap_cannot_explain_email_result():
    explainer = URLSHAPExplainer()

    result = AgentResult.from_probability(
        agent_name="email_analysis_agent",
        modality="email",
        phishing_probability=0.75,
        evidence=[],
        execution_metadata=None,
    )

    assert not explainer.can_explain(
        result,
        make_context(),
    )


def test_lime_can_explain():
    explainer = URLLIMEExplainer()

    assert explainer.can_explain(
        make_result(),
        make_context(),
    )


def test_lime_cannot_explain_without_features():
    explainer = URLLIMEExplainer()

    context = AgentContext(
        input_text="https://example.com",
        url="https://example.com",
        metadata={},
    )

    assert not explainer.can_explain(
        make_result(),
        context,
    )


def test_shap_requires_url_features():
    explainer = URLSHAPExplainer()

    context = AgentContext(
        input_text="https://example.com",
        url="https://example.com",
        metadata={},
    )

    with pytest.raises(ValueError):
        explainer.explain(
            result=make_result(),
            context=context,
        )


def test_lime_requires_url_features():
    explainer = URLLIMEExplainer()

    context = AgentContext(
        input_text="https://example.com",
        url="https://example.com",
        metadata={},
    )

    with pytest.raises(ValueError):
        explainer.explain(
            result=make_result(),
            context=context,
        )


def test_shap_explainer_metadata():
    explainer = URLSHAPExplainer()

    assert explainer.explainer_name == "url_shap_explainer"
    assert explainer.method == "shap"
    assert explainer.framework == "SHAP"


def test_lime_explainer_metadata():
    explainer = URLLIMEExplainer()

    assert explainer.explainer_name == "url_lime_explainer"
    assert explainer.method == "lime"
    assert explainer.framework == "LIME"


def test_combined_explainer():
    explainer = URLXAIExplainer()

    assert hasattr(explainer, "shap")
    assert hasattr(explainer, "lime")


def test_fake_shap_output_shape():
    model = FakeModel()
    explainer = FakeSHAPExplainer(model)

    values = np.array([[1.0, 2.0, 3.0]])

    output = explainer.shap_values(values)

    assert len(output) == 2
    assert output[1].shape == values.shape


def test_url_context_contains_features():
    context = make_context()

    assert "url_features" in context.metadata


def test_url_result_probability():
    result = make_result()

    assert result.phishing_probability == pytest.approx(0.75)


def test_url_result_modality():
    result = make_result()

    assert result.modality == "url"


def test_shap_scope():
    assert ExplanationScope.LOCAL.value == "local"


def test_explanation_direction_values():
    assert (
        ExplanationDirection.SUPPORTS_PHISHING.value
        == "supports_phishing"
    )

    assert (
        ExplanationDirection.SUPPORTS_LEGITIMATE.value
        == "supports_legitimate"
    )