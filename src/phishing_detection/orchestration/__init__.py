"""
Adaptive evidence orchestration.

Phase 3.8 introduces deterministic orchestration
across specialized phishing detection agents.
"""

from phishing_detection.orchestration.orchestrator import (
    OrchestrationResult,
    OrchestrationStep,
    OrchestratorAgent,
)

from phishing_detection.explainability.comparison import (
    XAIComparisonResult,
    compare_explanations,
)

__all__ = [
    "OrchestrationResult",
    "OrchestrationStep",
    "OrchestratorAgent",
]