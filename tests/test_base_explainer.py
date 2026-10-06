import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.base import (
    BaseExplainer,
    ExplanationMetadata,
    ExplanationResult,
)


class DummyExplainer(BaseExplainer):
    """Minimal concrete explainer used for testing."""

    def get_explainer_name(self) -> str:
        return "dummy_explainer"

    def get_method_name(self) -> str:
        return "dummy_method"

    def get_framework_name(self) -> str:
        return "DummyFramework"

    def explain(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> ExplanationResult:
        self.validate_inputs(result, context)

        metadata = self.create_metadata(
            model_name="dummy_model",
            execution_time_ms=1.5,
            additional_metadata={
                "phase": "4.1",
            },
        )

        return ExplanationResult(
            summary="Dummy explanation.",
            factors=[
                {
                    "feature": "dummy_feature",
                    "importance": 1.0,
                }
            ],
            metadata=metadata,
            prediction=result.phishing_probability,
        )


def make_result() -> AgentResult:
    return AgentResult.from_probability(
        agent_name="dummy_agent",
        modality="test",
        phishing_probability=0.9,
    )


def make_context() -> AgentContext:
    return AgentContext(
        input_text="This is a test email."
    )


def test_explanation_metadata_creation():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
        model_name="test_model",
        framework="SHAP",
        execution_time_ms=2.5,
    )

    assert metadata.explainer_name == "test_explainer"
    assert metadata.method == "test_method"
    assert metadata.model_name == "test_model"
    assert metadata.framework == "SHAP"
    assert metadata.execution_time_ms == 2.5


def test_explanation_metadata_to_dict():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
        framework="LIME",
    )

    result = metadata.to_dict()

    assert result["explainer_name"] == "test_explainer"
    assert result["method"] == "test_method"
    assert result["framework"] == "LIME"


def test_explanation_metadata_rejects_empty_name():
    with pytest.raises(ValueError):
        ExplanationMetadata(
            explainer_name="",
            method="test_method",
        )


def test_explanation_metadata_rejects_empty_method():
    with pytest.raises(ValueError):
        ExplanationMetadata(
            explainer_name="test_explainer",
            method="",
        )


def test_explanation_metadata_rejects_empty_framework():
    with pytest.raises(ValueError):
        ExplanationMetadata(
            explainer_name="test_explainer",
            method="test_method",
            framework="",
        )


def test_explanation_metadata_rejects_negative_execution_time():
    with pytest.raises(ValueError):
        ExplanationMetadata(
            explainer_name="test_explainer",
            method="test_method",
            execution_time_ms=-1,
        )


def test_explanation_result_creation():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
    )

    explanation = ExplanationResult(
        summary="Test explanation.",
        factors=[
            {
                "feature": "url_length",
                "importance": 0.8,
            }
        ],
        metadata=metadata,
        prediction=0.85,
    )

    assert explanation.summary == "Test explanation."
    assert explanation.factor_count == 1
    assert explanation.prediction == 0.85


def test_explanation_result_to_dict():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
    )

    explanation = ExplanationResult(
        summary="Test explanation.",
        factors=[],
        metadata=metadata,
        prediction=0.75,
    )

    result = explanation.to_dict()

    assert result["summary"] == "Test explanation."
    assert result["factors"] == []
    assert result["prediction"] == 0.75
    assert "metadata" in result
    assert "raw_explanation" not in result


def test_explanation_result_can_include_raw_explanation():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
    )

    explanation = ExplanationResult(
        summary="Test explanation.",
        factors=[],
        metadata=metadata,
        raw_explanation={
            "framework_value": 123,
        },
    )

    result = explanation.to_dict(
        include_raw_explanation=True
    )

    assert result["raw_explanation"]["framework_value"] == 123


def test_explanation_result_rejects_empty_summary():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
    )

    with pytest.raises(ValueError):
        ExplanationResult(
            summary="",
            factors=[],
            metadata=metadata,
        )


def test_explanation_result_rejects_invalid_factors():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
    )

    with pytest.raises(TypeError):
        ExplanationResult(
            summary="Test explanation.",
            factors="invalid",
            metadata=metadata,
        )


def test_explanation_result_rejects_invalid_prediction():
    metadata = ExplanationMetadata(
        explainer_name="test_explainer",
        method="test_method",
    )

    with pytest.raises(ValueError):
        ExplanationResult(
            summary="Test explanation.",
            factors=[],
            metadata=metadata,
            prediction=1.5,
        )


def test_base_explainer_name():
    explainer = DummyExplainer()

    assert explainer.get_explainer_name() == "dummy_explainer"


def test_base_explainer_method():
    explainer = DummyExplainer()

    assert explainer.get_method_name() == "dummy_method"


def test_base_explainer_framework():
    explainer = DummyExplainer()

    assert explainer.get_framework_name() == "DummyFramework"


def test_base_explainer_can_explain():
    explainer = DummyExplainer()

    result = make_result()
    context = make_context()

    assert explainer.can_explain(result, context) is True


def test_base_explainer_validation_rejects_invalid_result():
    explainer = DummyExplainer()
    context = make_context()

    with pytest.raises(TypeError):
        explainer.validate_inputs(
            result="invalid",
            context=context,
        )


def test_base_explainer_validation_rejects_invalid_context():
    explainer = DummyExplainer()
    result = make_result()

    with pytest.raises(TypeError):
        explainer.validate_inputs(
            result=result,
            context="invalid",
        )


def test_create_metadata():
    explainer = DummyExplainer()

    metadata = explainer.create_metadata(
        model_name="test_model",
        execution_time_ms=3.2,
        additional_metadata={
            "phase": "4.1",
        },
    )

    assert metadata.explainer_name == "dummy_explainer"
    assert metadata.method == "dummy_method"
    assert metadata.framework == "DummyFramework"
    assert metadata.model_name == "test_model"
    assert metadata.execution_time_ms == 3.2
    assert metadata.additional_metadata["phase"] == "4.1"


def test_dummy_explainer_generates_explanation():
    explainer = DummyExplainer()

    result = make_result()
    context = make_context()

    explanation = explainer.explain(
        result=result,
        context=context,
    )

    assert isinstance(explanation, ExplanationResult)
    assert explanation.summary == "Dummy explanation."
    assert explanation.factor_count == 1
    assert explanation.prediction == 0.9


def test_dummy_explainer_metadata():
    explainer = DummyExplainer()

    explanation = explainer.explain(
        result=make_result(),
        context=make_context(),
    )

    assert explanation.metadata.explainer_name == "dummy_explainer"
    assert explanation.metadata.method == "dummy_method"
    assert explanation.metadata.framework == "DummyFramework"
    assert explanation.metadata.model_name == "dummy_model"