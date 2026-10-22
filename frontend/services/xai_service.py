from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from phishing_detection.agents.base import AgentContext
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
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        """Generate individual explanations for each executed agent."""
        explanations: dict[str, Any] = {}

        for agent_res in agent_results:
            if agent_res is None:
                continue

            name = getattr(agent_res, "agent_name", "")
            explainer = self.integrator._select_explainer(name)

            if explainer is None:
                continue

            exp = None
            try:
                if context is not None:
                    try:
                        exp = explainer.explain(result=agent_res, context=context)
                    except TypeError:
                        exp = explainer.explain(agent_res, context)
                else:
                    exp = self.integrator.explain_agent_result(agent_res)
            except Exception as exc:
                logger.warning("Explainer for %s failed: %s", name, exc)
                continue

            if exp:
                normalized = self.integrator._normalize_explanation_result(exp)
                if normalized:
                    explanations[name] = normalized

        return explanations

    def build_explanation_result(
        self,
        agent_results: Sequence[AgentResult],
        decision_trace: Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build complete explainability output including per-agent
        and aggregated explanations.
        """
        metadata_dict = dict(metadata or {})
        context = AgentContext(
            input_text=metadata_dict.get("input_text"),
            url=metadata_dict.get("url"),
            sender=metadata_dict.get("sender"),
            metadata=metadata_dict,
        )

        # Convert OrchestrationStep objects to dicts so pipeline_integration's dict(item) succeeds
        serialized_trace = []
        for step in (decision_trace or []):
            if step is None:
                continue
            if hasattr(step, "to_dict"):
                serialized_trace.append(step.to_dict())
            elif isinstance(step, dict):
                serialized_trace.append(step)
            elif hasattr(step, "__dict__"):
                serialized_trace.append(
                    {k: v for k, v in vars(step).items() if not k.startswith("_")}
                )
            else:
                serialized_trace.append(dict(step))

        try:
            explanations = self.explain_agents(agent_results, context=context)
            pipeline_result = self.integrator.build_result(
                agent_results=agent_results,
                explanations=explanations,
                decision_trace=serialized_trace,
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
