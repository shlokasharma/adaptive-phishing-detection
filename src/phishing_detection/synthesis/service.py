"""
Evidence Synthesis Service for Phase 7 (Evidence Synthesis & LLM Reasoning).

Top-level service coordinating evidence collection, conflict resolution, risk assessment,
and explainability across Phase 3 orchestration, Phase 4 XAI, Phase 5 adaptive policy,
and Phase 6 RL.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from phishing_detection.agents.base import AgentContext
from phishing_detection.orchestration.orchestrator import OrchestrationResult
from phishing_detection.synthesis.collector import EvidenceCollector
from phishing_detection.synthesis.conflict_resolver import ConflictResolver
from phishing_detection.synthesis.explainer import DeterministicExplainer
from phishing_detection.synthesis.llm_adapter import (
    BaseLLMAdapter,
    CloudLLMAdapter,
    LLMConfig,
    NullLLMAdapter,
)
from phishing_detection.synthesis.schema import (
    ConflictReport,
    RiskLevel,
    StandardizedEvidence,
    SynthesisAssessment,
)
from phishing_detection.synthesis.synthesizer import EvidenceSynthesizer

logger = logging.getLogger(__name__)


class EvidenceSynthesisService:
    """
    Unified Phase 7 service for synthesizing evidence and generating grounded explanations.
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        llm_adapter: BaseLLMAdapter | None = None,
        synthesizer: EvidenceSynthesizer | None = None,
    ) -> None:
        if synthesizer is not None:
            self.synthesizer = synthesizer
        else:
            adapter = llm_adapter
            if adapter is None:
                if llm_config is not None and llm_config.enabled:
                    adapter = CloudLLMAdapter(config=llm_config)
                else:
                    adapter = NullLLMAdapter()

            self.synthesizer = EvidenceSynthesizer(
                collector=EvidenceCollector(),
                conflict_resolver=ConflictResolver(),
                llm_adapter=adapter,
            )

    def synthesize(
        self,
        orchestration_result: OrchestrationResult,
        context: AgentContext | None = None,
        xai_explanations: Sequence[Any] | None = None,
    ) -> SynthesisAssessment:
        """
        Synthesize orchestration results into a grounded, explainable assessment.
        """
        return self.synthesizer.synthesize(
            orchestration_result=orchestration_result,
            context=context,
            xai_explanations=xai_explanations,
        )

    def enrich_pipeline_result(
        self,
        pipeline_result_dict: dict[str, Any],
        orchestration_result: OrchestrationResult,
        context: AgentContext | None = None,
        xai_explanations: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Enrich an existing pipeline dictionary with Phase 7 synthesis fields.

        Maintains 100% backward compatibility with existing keys ('label', 'label_name',
        'confidence', 'evidence', 'decision_trace') while adding 'synthesis' section.
        """
        assessment = self.synthesize(
            orchestration_result=orchestration_result,
            context=context,
            xai_explanations=xai_explanations,
        )

        enriched = dict(pipeline_result_dict)
        enriched["synthesis"] = assessment.to_dict()
        enriched["risk_level"] = assessment.risk_level.value if hasattr(assessment.risk_level, "value") else str(assessment.risk_level)
        enriched["final_verdict"] = assessment.final_verdict
        enriched["explanation"] = assessment.explanation
        enriched["decision_summary"] = assessment.summary
        enriched["conflict_detected"] = assessment.conflict_report.has_conflict

        return enriched
