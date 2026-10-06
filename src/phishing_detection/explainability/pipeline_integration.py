"""
Phase 4.8 - Pipeline XAI Integration.

This module integrates the Phase 3 multi-agent phishing detection pipeline
with the Phase 4 explainability layer.

Architecture
------------
Phase 3 AgentResult objects
        |
        v
ExplainabilityPipelineIntegrator
        |
        +--> Email XAI
        +--> URL XAI
        +--> Sender native XAI
        |
        v
MultiAgentExplanationAggregator
        |
        v
ExplainablePipelineResult

The integration layer deliberately does not modify the Phase 3 pipeline.
This keeps Phase 3 behaviour frozen while allowing explainability to be
attached as an independent layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from phishing_detection.agents.result import AgentResult

from phishing_detection.explainability.aggregation import (
    MultiAgentExplanationAggregator,
)
from phishing_detection.explainability.email_xai import (
    EmailXAIExplainer,
)
from phishing_detection.explainability.schema import UnifiedExplanation
from phishing_detection.explainability.sender_xai import (
    SenderXAIExplainer,
)
from phishing_detection.explainability.url_xai import (
    URLXAIExplainer,
)


@dataclass
class ExplainablePipelineResult:
    """
    Final result produced by the Phase 4.8 integration layer.

    Parameters
    ----------
    agent_results:
        Agent results produced by the Phase 3 pipeline.

    explanations:
        Per-agent explanations. The mapping is keyed by normalized agent name.

    aggregated_explanation:
        Combined multi-agent explanation.

    final_agent_result:
        Agent result selected as the final decision source.

    decision:
        Final human-readable decision.

    confidence:
        Final confidence.

    phishing_probability:
        Final phishing probability.

    decision_trace:
        Phase 3 decision trace, preserved without modification.

    metadata:
        Additional integration metadata.
    """

    agent_results: list[AgentResult] = field(default_factory=list)
    explanations: dict[str, dict[str, UnifiedExplanation]] = field(
        default_factory=dict
    )
    aggregated_explanation: UnifiedExplanation | None = None
    final_agent_result: AgentResult | None = None
    decision: str | None = None
    confidence: float | None = None
    phishing_probability: float | None = None
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def explanation(self) -> UnifiedExplanation | None:
        """Backward-compatible alias for the aggregated explanation."""
        return self.aggregated_explanation

    @property
    def final_explanation(self) -> UnifiedExplanation | None:
        """Return the aggregated final explanation."""
        return self.aggregated_explanation

    @property
    def agent_explanations(self) -> dict[str, dict[str, UnifiedExplanation]]:
        """Return per-agent explanations."""
        return self.explanations

    @property
    def is_explainable(self) -> bool:
        """Whether at least one explanation is available."""
        return self.aggregated_explanation is not None or bool(self.explanations)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the complete explainable pipeline result.
        """

        return {
            "agent_results": [
                self._agent_result_to_dict(result)
                for result in self.agent_results
            ],
            "explanations": {
                agent_name: {
                    method: explanation.to_dict()
                    for method, explanation in method_explanations.items()
                }
                for agent_name, method_explanations in self.explanations.items()
            },
            "aggregated_explanation": (
                self.aggregated_explanation.to_dict()
                if self.aggregated_explanation is not None
                else None
            ),
            "final_agent_result": (
                self._agent_result_to_dict(self.final_agent_result)
                if self.final_agent_result is not None
                else None
            ),
            "decision": self.decision,
            "confidence": self.confidence,
            "phishing_probability": self.phishing_probability,
            "decision_trace": self.decision_trace,
            "metadata": self.metadata,
        }

    @staticmethod
    def _agent_result_to_dict(
        result: AgentResult | None,
    ) -> dict[str, Any] | None:
        """Serialize an AgentResult without requiring a to_dict method."""

        if result is None:
            return None

        execution_metadata = getattr(result, "execution_metadata", None)

        if execution_metadata is None:
            execution_metadata_dict = None
        elif hasattr(execution_metadata, "to_dict"):
            execution_metadata_dict = execution_metadata.to_dict()
        else:
            execution_metadata_dict = getattr(
                execution_metadata,
                "__dict__",
                execution_metadata,
            )

        return {
            "agent_name": getattr(result, "agent_name", None),
            "modality": getattr(result, "modality", None),
            "label": getattr(result, "label", None),
            "label_name": getattr(result, "label_name", None),
            "phishing_probability": getattr(
                result,
                "phishing_probability",
                None,
            ),
            "confidence": getattr(result, "confidence", None),
            "uncertainty": getattr(result, "uncertainty", None),
            "evidence": [
                (
                    evidence.to_dict()
                    if hasattr(evidence, "to_dict")
                    else getattr(evidence, "__dict__", evidence)
                )
                for evidence in getattr(result, "evidence", [])
            ],
            "execution_metadata": execution_metadata_dict,
            "metadata": dict(getattr(result, "metadata", {}) or {}),
        }


class ExplainabilityPipelineIntegrator:
    """
    Integrate Phase 4 explainability with Phase 3 AgentResult objects.

    The class owns one XAI explainer per supported modality:

    * EmailAnalysisAgent -> EmailXAIExplainer
    * URLAnalysisAgent -> URLXAIExplainer
    * SenderAnalysisAgent -> SenderXAIExplainer

    The implementation is intentionally independent from the Phase 3
    orchestrator so the later reinforcement-learning policy can replace the
    deterministic policy without rewriting the XAI layer.
    """

    SUPPORTED_AGENTS = {
        "emailanalysisagent": "email",
        "email_analysis_agent": "email",
        "email": "email",
        "urlanalysisagent": "url",
        "url_analysis_agent": "url",
        "url": "url",
        "senderanalysisagent": "sender",
        "sender_analysis_agent": "sender",
        "sender": "sender",
    }

    def __init__(
        self,
        email_explainer: EmailXAIExplainer | None = None,
        url_explainer: URLXAIExplainer | None = None,
        sender_explainer: SenderXAIExplainer | None = None,
        aggregator: MultiAgentExplanationAggregator | None = None,
    ) -> None:
        """
        Construct the integration layer.

        Optional explainer injection makes the component easy to test and
        allows callers to provide preconfigured explainers.
        """

        self.email_explainer = (
            email_explainer
            if email_explainer is not None
            else EmailXAIExplainer()
        )

        self.url_explainer = (
            url_explainer
            if url_explainer is not None
            else URLXAIExplainer()
        )

        self.sender_explainer = (
            sender_explainer
            if sender_explainer is not None
            else SenderXAIExplainer()
        )

        self.aggregator = (
            aggregator
            if aggregator is not None
            else MultiAgentExplanationAggregator()
        )

    # ------------------------------------------------------------------
    # Agent handling
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_agent_name(cls, agent_name: str) -> str:
        """Normalize agent names for consistent routing."""

        if not agent_name:
            return ""

        normalized = (
            str(agent_name)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return normalized

    @classmethod
    def _agent_modality(cls, agent_name: str) -> str | None:
        """Return the supported modality for an agent."""

        normalized = cls._normalize_agent_name(agent_name)

        return cls.SUPPORTED_AGENTS.get(normalized)

    def _select_explainer(
        self,
        agent_name: str,
    ) -> Any | None:
        """Select the XAI explainer corresponding to an agent."""

        modality = self._agent_modality(agent_name)

        if modality == "email":
            return self.email_explainer

        if modality == "url":
            return self.url_explainer

        if modality == "sender":
            return self.sender_explainer

        return None

    # ------------------------------------------------------------------
    # Explanation normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_explanation_result(
        result: Any,
    ) -> dict[str, UnifiedExplanation]:
        """
        Normalize any supported explainer output into a dictionary.

        Supported forms
        ----------------
        UnifiedExplanation
            -> {"explanation": explanation}

        dict
            {"shap": UnifiedExplanation, "lime": UnifiedExplanation}

        None
            -> {}

        Nested dictionaries are recursively flattened while preserving
        recognized SHAP/LIME/native explanation method names.
        """

        if result is None:
            return {}

        if isinstance(result, UnifiedExplanation):
            method = (
                str(getattr(result, "method", "") or "")
                .strip()
                .lower()
            )

            if method in {"shap", "lime", "rule_based", "native"}:
                key = method
            else:
                key = "explanation"

            return {key: result}

        if isinstance(result, Mapping):
            normalized: dict[str, UnifiedExplanation] = {}

            for key, value in result.items():
                key_string = str(key).strip().lower()

                if isinstance(value, UnifiedExplanation):
                    normalized[key_string] = value
                    continue

                if isinstance(value, Mapping):
                    nested = ExplainabilityPipelineIntegrator._normalize_explanation_result(
                        value
                    )

                    for nested_key, nested_value in nested.items():
                        if key_string in {
                            "shap",
                            "lime",
                            "rule_based",
                            "native",
                        }:
                            normalized[key_string] = nested_value
                        else:
                            normalized[nested_key] = nested_value

            return normalized

        if isinstance(result, Sequence) and not isinstance(
            result,
            (str, bytes, bytearray),
        ):
            normalized = {}

            for index, item in enumerate(result):
                nested = ExplainabilityPipelineIntegrator._normalize_explanation_result(
                    item
                )

                for key, value in nested.items():
                    normalized[f"{key}_{index}"] = value

            return normalized

        return {}

    # ------------------------------------------------------------------
    # Explainer invocation
    # ------------------------------------------------------------------

    @staticmethod
    def _call_explainer(
        explainer: Any,
        agent_result: AgentResult,
    ) -> Any:
        """
        Invoke an explainer while supporting the existing Phase 4 explainer
        interfaces.

        The function first attempts the public ``explain`` method and then
        falls back to modality-specific callable interfaces if required.
        """

        explain_method = getattr(explainer, "explain", None)

        if callable(explain_method):
            attempts = (
                lambda: explain_method(agent_result),
                lambda: explain_method(result=agent_result),
                lambda: explain_method(agent_result=agent_result),
            )

            last_error: Exception | None = None

            for attempt in attempts:
                try:
                    return attempt()
                except TypeError as exc:
                    last_error = exc

            if last_error is not None:
                raise last_error

        if callable(explainer):
            return explainer(agent_result)

        raise TypeError(
            f"Explainer {type(explainer).__name__} does not expose "
            "a supported explain interface."
        )

    # ------------------------------------------------------------------
    # Per-agent explanation
    # ------------------------------------------------------------------

    def explain_agent_result(
        self,
        agent_result: AgentResult,
    ) -> dict[str, UnifiedExplanation]:
        """
        Explain one Phase 3 AgentResult.

        Unknown agents are deliberately ignored rather than raising an
        exception. This makes the integration forward-compatible with
        future specialized agents.
        """

        if agent_result is None:
            return {}

        agent_name = getattr(agent_result, "agent_name", "")

        explainer = self._select_explainer(agent_name)

        if explainer is None:
            return {}

        raw_result = self._call_explainer(
            explainer,
            agent_result,
        )

        return self._normalize_explanation_result(raw_result)

    # Backward-compatible alias.
    explain = explain_agent_result

    def explain_agents(
        self,
        agent_results: Iterable[AgentResult],
    ) -> dict[str, dict[str, UnifiedExplanation]]:
        """
        Explain all supported AgentResult objects.

        Returns
        -------
        dict
            {
                "EmailAnalysisAgent": {
                    "shap": UnifiedExplanation,
                    "lime": UnifiedExplanation,
                },
                ...
            }
        """

        explanations: dict[str, dict[str, UnifiedExplanation]] = {}

        for agent_result in agent_results:
            if agent_result is None:
                continue

            agent_name = str(
                getattr(agent_result, "agent_name", "")
            ).strip()

            if not agent_name:
                continue

            normalized = self._normalize_agent_name(agent_name)

            if self._agent_modality(normalized) is None:
                continue

            result = self.explain_agent_result(agent_result)

            if result:
                explanations[agent_name] = result

        return explanations

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_explanations(
        explanations: Mapping[
            str,
            Mapping[str, UnifiedExplanation],
        ]
        | Iterable[UnifiedExplanation]
        | UnifiedExplanation,
    ) -> list[UnifiedExplanation]:
        """
        Flatten nested agent/method explanation structures.

        The Phase 4 aggregator itself expects an iterable of
        UnifiedExplanation objects, while the pipeline integration layer
        exposes the more useful per-agent nested structure.
        """

        if explanations is None:
            return []

        if isinstance(explanations, UnifiedExplanation):
            return [explanations]

        if isinstance(explanations, Mapping):
            flattened: list[UnifiedExplanation] = []

            for value in explanations.values():
                if isinstance(value, UnifiedExplanation):
                    flattened.append(value)

                elif isinstance(value, Mapping):
                    for nested_value in value.values():
                        if isinstance(nested_value, UnifiedExplanation):
                            flattened.append(nested_value)

            return flattened

        flattened = []

        for item in explanations:
            if isinstance(item, UnifiedExplanation):
                flattened.append(item)

            elif isinstance(item, Mapping):
                flattened.extend(
                    ExplainabilityPipelineIntegrator._flatten_explanations(
                        item
                    )
                )

        return flattened

    def aggregate(
        self,
        explanations: Mapping[
            str,
            Mapping[str, UnifiedExplanation],
        ]
        | Iterable[UnifiedExplanation]
        | UnifiedExplanation,
    ) -> UnifiedExplanation | None:
        """
        Aggregate explanations from one or multiple agents.

        Both of the following are supported:

        1. Flat:
           [UnifiedExplanation, UnifiedExplanation]

        2. Nested:
           {
               "EmailAnalysisAgent": {
                   "shap": UnifiedExplanation,
                   "lime": UnifiedExplanation,
               }
           }
        """

        flattened = self._flatten_explanations(explanations)

        if not flattened:
            return None

        return self.aggregator.aggregate(flattened)

    # ------------------------------------------------------------------
    # Final result construction
    # ------------------------------------------------------------------

    @staticmethod
    def _select_final_agent_result(
        agent_results: Sequence[AgentResult],
    ) -> AgentResult | None:
        """
        Select the final AgentResult.

        Phase 3's orchestrator already determines execution order and final
        decision. Therefore this layer preserves the last produced result
        rather than introducing a second decision policy.
        """

        if not agent_results:
            return None

        return agent_results[-1]

    @staticmethod
    def _extract_decision(
        agent_result: AgentResult | None,
    ) -> str | None:
        """Extract a human-readable decision from an AgentResult."""

        if agent_result is None:
            return None

        label_name = getattr(agent_result, "label_name", None)

        if label_name:
            return str(label_name)

        label = getattr(agent_result, "label", None)

        if label is None:
            return None

        if isinstance(label, bool):
            return "phishing" if label else "legitimate"

        if isinstance(label, int):
            return "phishing" if label == 1 else "legitimate"

        return str(label)

    def build_result(
        self,
        agent_results: Sequence[AgentResult],
        explanations: Mapping[
            str,
            Mapping[str, UnifiedExplanation],
        ]
        | None = None,
        decision_trace: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExplainablePipelineResult:
        """
        Build the final ExplainablePipelineResult.
        """

        results = list(agent_results)

        final_agent_result = self._select_final_agent_result(
            results
        )

        if explanations is None:
            explanations = {}

        aggregated = self.aggregate(explanations)

        phishing_probability = (
            getattr(
                final_agent_result,
                "phishing_probability",
                None,
            )
            if final_agent_result is not None
            else None
        )

        confidence = (
            getattr(
                final_agent_result,
                "confidence",
                None,
            )
            if final_agent_result is not None
            else None
        )

        final_metadata = dict(metadata or {})

        final_metadata.setdefault(
            "phase",
            "4.8",
        )

        final_metadata.setdefault(
            "integration_type",
            "pipeline_xai_integration",
        )

        final_metadata.setdefault(
            "agent_count",
            len(results),
        )

        final_metadata.setdefault(
            "supported_agent_count",
            sum(
                1
                for result in results
                if self._agent_modality(
                    getattr(result, "agent_name", "")
                )
                is not None
            ),
        )

        return ExplainablePipelineResult(
            agent_results=results,
            explanations=dict(explanations),
            aggregated_explanation=aggregated,
            final_agent_result=final_agent_result,
            decision=self._extract_decision(
                final_agent_result
            ),
            confidence=confidence,
            phishing_probability=phishing_probability,
            decision_trace=[
                dict(item)
                for item in (decision_trace or [])
            ],
            metadata=final_metadata,
        )

    # ------------------------------------------------------------------
    # Complete integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        agent_results: Sequence[AgentResult],
        decision_trace: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExplainablePipelineResult:
        """
        Run the complete Phase 4.8 explainability integration.
        """

        start = perf_counter()

        results = list(agent_results)

        explanations = self.explain_agents(results)

        result_metadata = dict(metadata or {})

        result_metadata["integration_execution_time_ms"] = round(
            (perf_counter() - start) * 1000.0,
            4,
        )

        return self.build_result(
            agent_results=results,
            explanations=explanations,
            decision_trace=decision_trace,
            metadata=result_metadata,
        )


__all__ = [
    "ExplainabilityPipelineIntegrator",
    "ExplainablePipelineResult",
]