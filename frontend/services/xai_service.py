from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.pipeline_integration import (
    ExplainabilityPipelineIntegrator,
    ExplainablePipelineResult,
)

logger = logging.getLogger(__name__)


class XAIService:
    """
    Frontend service wrapper for Phase 4 explainability integration.

    Connects the Streamlit UI to SHAP, LIME, and rule-based explainers
    via ExplainabilityPipelineIntegrator.
    """

    def __init__(
        self,
        integrator: ExplainabilityPipelineIntegrator | None = None,
    ) -> None:
        self.integrator = (
            integrator
            if integrator is not None
            else ExplainabilityPipelineIntegrator()
        )

    def explain_agents(
        self,
        agent_results: Sequence[AgentResult],
    ) -> dict[str, Any]:
        """Generate individual explanations for each executed agent."""
        return self.integrator.explain_agents(agent_results)

    def build_explanation_result(
        self,
        agent_results: Sequence[AgentResult],
        decision_trace: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build complete explainability output including per-agent
        and aggregated explanations.
        """
        try:
            explanations = self.integrator.explain_agents(agent_results)
            pipeline_result = self.integrator.build_result(
                agent_results=agent_results,
                explanations=explanations,
                decision_trace=decision_trace,
                metadata=metadata,
            )
            return pipeline_result.to_dict()
        except Exception as exc:
            logger.warning("XAI explanation generation failed: %s", exc)
            return {
                "error": str(exc),
                "explanations": {},
                "aggregated_explanation": None,
            }
