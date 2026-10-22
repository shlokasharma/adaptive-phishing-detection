"""
Evidence Synthesizer for Phase 7 (Evidence Synthesis & LLM Reasoning).

Fuses normalized multi-agent evidence, executes conflict resolution, determines final
risk level, and coordinates grounded natural-language explanations.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from phishing_detection.agents.base import AgentContext
from phishing_detection.orchestration.orchestrator import OrchestrationResult
from phishing_detection.synthesis.collector import EvidenceCollector
from phishing_detection.synthesis.conflict_resolver import ConflictResolver
from phishing_detection.synthesis.explainer import DeterministicExplainer
from phishing_detection.synthesis.llm_adapter import BaseLLMAdapter, NullLLMAdapter
from phishing_detection.synthesis.schema import (
    ConflictReport,
    EvidenceCategory,
    EvidencePolarity,
    RiskLevel,
    StandardizedEvidence,
    SynthesisAssessment,
)

logger = logging.getLogger(__name__)


class EvidenceSynthesizer:
    """
    Core synthesis engine for Phase 7.
    """

    def __init__(
        self,
        collector: EvidenceCollector | None = None,
        conflict_resolver: ConflictResolver | None = None,
        llm_adapter: BaseLLMAdapter | None = None,
        uncertainty_threshold: float = 0.40,
    ) -> None:
        self.collector = collector or EvidenceCollector()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.llm_adapter = llm_adapter or NullLLMAdapter()
        self.uncertainty_threshold = uncertainty_threshold

    def synthesize(
        self,
        orchestration_result: OrchestrationResult,
        context: AgentContext | None = None,
        xai_explanations: Sequence[Any] | None = None,
    ) -> SynthesisAssessment:
        """
        Produce a complete, explainable SynthesisAssessment from an OrchestrationResult.
        """
        # 1. Collect and normalize evidence
        evidence_items, missing_components, failed_components = self.collector.collect_from_orchestration(
            orchestration_result=orchestration_result,
            context=context,
            xai_explanations=xai_explanations,
        )

        # 2. Analyze conflicts
        conflict_report = self.conflict_resolver.analyze_conflicts(evidence_items)

        # 3. Categorize evidence counts
        phish_items = [e for e in evidence_items if e.polarity == EvidencePolarity.SUPPORTS_PHISHING]
        legit_items = [e for e in evidence_items if e.polarity == EvidencePolarity.SUPPORTS_LEGITIMATE]
        inconclusive_items = [
            e for e in evidence_items
            if e.polarity in (EvidencePolarity.INCONCLUSIVE, EvidencePolarity.NEUTRAL)
        ]

        # 4. Compute synthesized confidence and uncertainty
        base_confidence = float(getattr(orchestration_result, "confidence", 0.5))
        # Apply conflict discount
        adj_confidence = max(0.0, min(1.0, base_confidence - conflict_report.confidence_discount))
        uncertainty = round(1.0 - adj_confidence, 4)

        # 5. Determine final verdict and risk level
        base_label = int(getattr(orchestration_result, "label", 0))

        # Check for unresolved severe conflict
        if conflict_report.has_conflict and uncertainty >= self.uncertainty_threshold:
            verdict = "uncertain"
            risk_level = RiskLevel.UNCERTAIN
            binary_label = base_label  # Preserve orchestrator's decision for backward compatibility
        elif base_label == 1:
            verdict = "phishing"
            if adj_confidence >= 0.85:
                risk_level = RiskLevel.CRITICAL if len(phish_items) >= 2 else RiskLevel.HIGH
            elif adj_confidence >= 0.65:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.MEDIUM
            binary_label = 1
        else:
            verdict = "legitimate"
            if adj_confidence >= 0.85:
                risk_level = RiskLevel.LOW
            elif adj_confidence >= 0.60:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.MEDIUM
            binary_label = 0

        # Special case: If an agent failed and remaining evidence is weak, mark uncertain
        if failed_components and len(phish_items) == 0 and len(legit_items) == 0:
            verdict = "uncertain"
            risk_level = RiskLevel.UNCERTAIN

        # 6. Generate natural language reasoning explanation
        summary, explanation, explanation_source = self.llm_adapter.generate_explanation(
            final_verdict=verdict,
            risk_level=risk_level,
            confidence=adj_confidence,
            uncertainty=uncertainty,
            evidence_items=evidence_items,
            conflict_report=conflict_report,
            missing_components=missing_components,
            failed_components=failed_components,
        )

        metadata = {
            "orchestrator_label": base_label,
            "orchestrator_confidence": base_confidence,
            "conflict_detected": conflict_report.has_conflict,
            "confidence_discount": conflict_report.confidence_discount,
            "total_evidence_collected": len(evidence_items),
        }

        return SynthesisAssessment(
            final_verdict=verdict,
            binary_label=binary_label,
            risk_level=risk_level,
            overall_confidence=round(adj_confidence, 4),
            uncertainty=uncertainty,
            supporting_phishing_count=len(phish_items),
            supporting_legitimate_count=len(legit_items),
            inconclusive_count=len(inconclusive_items),
            missing_evidence_count=len(missing_components),
            failed_components_count=len(failed_components),
            conflict_report=conflict_report,
            summary=summary,
            explanation=explanation,
            explanation_source=explanation_source,
            evidence_items=evidence_items,
            missing_components=missing_components,
            failed_components=failed_components,
            synthesis_metadata=metadata,
        )
