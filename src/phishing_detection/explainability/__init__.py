"""
Explainability package for adaptive phishing detection.
"""

from phishing_detection.explainability.base import (
    BaseExplainer,
    ExplanationMetadata,
    ExplanationResult,
)

from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)

from phishing_detection.explainability.email_xai import (
    EmailLIMEExplainer,
    EmailSHAPExplainer,
    EmailXAIExplainer,
)

from phishing_detection.explainability.url_xai import (
    URLLIMEExplainer,
    URLSHAPExplainer,
    URLXAIExplainer,
)

from phishing_detection.explainability.sender_xai import (
    SenderXAIExplainer,
)

from phishing_detection.explainability.comparison import (
    XAIComparisonResult,
    compare_explanations,
)

from phishing_detection.explainability.aggregation import (
    AggregatedFactor,
    MultiAgentExplanationAggregator,
)

from phishing_detection.explainability.pipeline_integration import (
    ExplainabilityPipelineIntegrator,
    ExplainablePipelineResult,
)

from phishing_detection.explainability.evaluation import (
    AgreementResult,
    ConsistencyResult,
    FidelityResult,
    RobustnessResult,
    RuntimeResult,
    SparsityResult,
    XAIQualityEvaluator,
    XAIQualityReport,
)

__all__ = [
    "BaseExplainer",
    "ExplanationMetadata",
    "ExplanationResult",
    "ExplanationDirection",
    "ExplanationFactor",
    "ExplanationScope",
    "ExplanationTarget",
    "UnifiedExplanation",
    "EmailSHAPExplainer",
    "EmailLIMEExplainer",
    "EmailXAIExplainer",
    "URLSHAPExplainer",
    "URLLIMEExplainer",
    "URLXAIExplainer",
    "SenderXAIExplainer",
    "XAIComparisonResult",
    "compare_explanations",
    "AggregatedFactor",
    "MultiAgentExplanationAggregator",
    "ExplainabilityPipelineIntegrator",
    "ExplainablePipelineResult",
]