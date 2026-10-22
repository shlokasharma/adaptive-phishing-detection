"""
Evidence Collector and Normalizer for Phase 7 (Evidence Synthesis & LLM Reasoning).

Collects heterogeneous outputs from OrchestrationResult, AgentResult, Evidence items,
and Phase 4 XAI Explanations. Normalizes them into StandardizedEvidence with duplicate
detection, provenance tracking, and missing/failed component identification.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Sequence

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.orchestration.orchestrator import OrchestrationResult, OrchestrationStep
from phishing_detection.synthesis.schema import (
    EvidenceCategory,
    EvidencePolarity,
    EvidenceStatus,
    ScoreType,
    StandardizedEvidence,
)

logger = logging.getLogger(__name__)


def _compute_finding_hash(source: str, category: str, finding_text: str) -> str:
    """Create a unique deterministic hash for deduplication."""
    norm = f"{source.strip().lower()}:{category.strip().lower()}:{finding_text.strip().lower()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def _map_agent_modality(modality: str) -> EvidenceCategory:
    """Map modality string to EvidenceCategory."""
    mod = (modality or "").lower()
    if "email" in mod or "text" in mod:
        return EvidenceCategory.EMAIL
    elif "url" in mod:
        return EvidenceCategory.URL
    elif "sender" in mod or "header" in mod:
        return EvidenceCategory.SENDER
    elif "transformer" in mod:
        return EvidenceCategory.TRANSFORMER
    elif "ml" in mod or "svm" in mod or "forest" in mod:
        return EvidenceCategory.TRADITIONAL_ML
    elif "threat" in mod or "intel" in mod or "reputation" in mod:
        return EvidenceCategory.THREAT_INTELLIGENCE
    return EvidenceCategory.UNKNOWN


class EvidenceCollector:
    """
    Normalizes and aggregates evidence across agents, models, and explainers.
    """

    KNOWN_COMPONENTS = [
        "email_analysis_agent",
        "url_analysis_agent",
        "sender_analysis_agent",
    ]

    def __init__(self, deduplicate: bool = True) -> None:
        self.deduplicate = deduplicate

    def collect_from_orchestration(
        self,
        orchestration_result: OrchestrationResult,
        context: AgentContext | None = None,
        xai_explanations: Sequence[Any] | None = None,
    ) -> tuple[list[StandardizedEvidence], list[str], list[dict[str, Any]]]:
        """
        Collect standardized evidence from an OrchestrationResult.

        Returns
        -------
        tuple[list[StandardizedEvidence], list[str], list[dict[str, Any]]]
            (collected_evidence, missing_components, failed_components)
        """
        collected: list[StandardizedEvidence] = []
        seen_hashes: set[str] = set()
        missing_components: list[str] = []
        failed_components: list[dict[str, Any]] = []

        # 1. Track executed agents and failures from decision trace
        executed_agent_names = set()
        if hasattr(orchestration_result, "decision_trace"):
            for step in orchestration_result.decision_trace:
                if isinstance(step, OrchestrationStep):
                    if step.action in ("execute", "execute_rl"):
                        executed_agent_names.add(step.agent_name)
                    elif step.action == "error":
                        failed_components.append({
                            "component": step.agent_name,
                            "reason": step.reason,
                            "step_number": step.step_number,
                        })

        # 2. Extract evidence from AgentResult objects
        agent_results = getattr(orchestration_result, "agent_results", [])
        for res in agent_results:
            if not isinstance(res, AgentResult):
                continue
            executed_agent_names.add(res.agent_name)
            agent_evidence = self._collect_from_agent_result(res)
            for item in agent_evidence:
                f_hash = _compute_finding_hash(item.source, item.category.value, item.finding)
                if self.deduplicate and f_hash in seen_hashes:
                    continue
                seen_hashes.add(f_hash)
                collected.append(item)

        # 3. Extract standalone evidence items from orchestration_result.evidence
        standalone_evidence = getattr(orchestration_result, "evidence", [])
        for ev in standalone_evidence:
            std_ev = self._normalize_raw_evidence(ev)
            if std_ev is not None:
                f_hash = _compute_finding_hash(std_ev.source, std_ev.category.value, std_ev.finding)
                if self.deduplicate and f_hash in seen_hashes:
                    continue
                seen_hashes.add(f_hash)
                collected.append(std_ev)

        # 4. Extract XAI explanation factors if provided
        if xai_explanations:
            xai_items = self._collect_from_xai(xai_explanations)
            for item in xai_items:
                f_hash = _compute_finding_hash(item.source, item.category.value, item.finding)
                if self.deduplicate and f_hash in seen_hashes:
                    continue
                seen_hashes.add(f_hash)
                collected.append(item)

        # 5. Check for missing known components
        for comp in self.KNOWN_COMPONENTS:
            if comp not in executed_agent_names and not any(f["component"] == comp for f in failed_components):
                # Check why it was missing
                reason = "Not executed (orchestration policy satisfied early or budget exhausted)"
                if comp == "url_analysis_agent" and context is not None and not bool(context.url):
                    reason = "No URL present in analyzed input"
                elif comp == "sender_analysis_agent" and context is not None and not bool(context.sender):
                    reason = "No sender/headers provided in analyzed input"
                missing_components.append(f"{comp}: {reason}")

        # Always check transformer component status
        missing_components.append("transformer_detector: Not configured/offline in current environment")

        return collected, missing_components, failed_components

    def _collect_from_agent_result(self, res: AgentResult) -> list[StandardizedEvidence]:
        """Convert an AgentResult and its internal Evidence items."""
        items: list[StandardizedEvidence] = []
        category = _map_agent_modality(res.modality)

        # Main agent prediction finding
        pred_label_str = "phishing" if res.label == 1 else "legitimate"
        polarity = EvidencePolarity.SUPPORTS_PHISHING if res.label == 1 else EvidencePolarity.SUPPORTS_LEGITIMATE
        finding_text = (
            f"{res.agent_name} classified content as {pred_label_str} "
            f"(probability={res.phishing_probability:.4f}, confidence={res.confidence:.4f})"
        )

        provenance = {
            "source_type": "agent_result",
            "modality": res.modality,
            "agent_name": res.agent_name,
        }
        if res.execution_metadata:
            provenance["execution_time_ms"] = res.execution_metadata.execution_time_ms
            provenance["model_name"] = res.execution_metadata.model_name

        items.append(
            StandardizedEvidence(
                source=res.agent_name,
                category=category,
                finding=finding_text,
                polarity=polarity,
                prediction=res.label,
                score=res.phishing_probability,
                score_type=ScoreType.PROBABILITY,
                confidence=res.confidence,
                supporting_explanation=f"Agent produced {pred_label_str} with {res.confidence:.1%} confidence.",
                provenance=provenance,
                metadata=res.metadata,
                status=EvidenceStatus.AVAILABLE,
            )
        )

        # Internal evidence items
        for ev in res.evidence:
            std = self._normalize_raw_evidence(ev, default_source=res.agent_name, default_category=category)
            if std is not None:
                items.append(std)

        return items

    def _normalize_raw_evidence(
        self,
        ev: Any,
        default_source: str = "orchestration",
        default_category: EvidenceCategory = EvidenceCategory.ORCHESTRATION,
    ) -> StandardizedEvidence | None:
        """Convert a Phase 3 Evidence object into StandardizedEvidence."""
        if isinstance(ev, StandardizedEvidence):
            return ev

        if not isinstance(ev, Evidence):
            return None

        # Determine polarity
        polarity = EvidencePolarity.NEUTRAL
        if ev.polarity == "supports_phishing":
            polarity = EvidencePolarity.SUPPORTS_PHISHING
        elif ev.polarity == "supports_legitimate":
            polarity = EvidencePolarity.SUPPORTS_LEGITIMATE

        category = _map_agent_modality(ev.evidence_type)
        if category == EvidenceCategory.UNKNOWN:
            category = default_category

        # Distinguish score type
        score = None
        score_type = ScoreType.UNKNOWN
        if isinstance(ev.value, (int, float)):
            score = float(ev.value)
            score_type = ScoreType.RISK_SCORE
            if 0.0 <= score <= 1.0 and "prob" in ev.description.lower():
                score_type = ScoreType.PROBABILITY

        return StandardizedEvidence(
            source=ev.source or default_source,
            category=category,
            finding=ev.description,
            polarity=polarity,
            score=score,
            score_type=score_type,
            confidence=ev.confidence,
            supporting_explanation=f"Finding from {ev.source}: {ev.description}",
            provenance={"evidence_type": ev.evidence_type, "reliability": ev.reliability},
            metadata=dict(ev.metadata),
            status=EvidenceStatus.AVAILABLE,
        )

    def _collect_from_xai(self, explanations: Sequence[Any]) -> list[StandardizedEvidence]:
        """Convert Phase 4 UnifiedExplanation and ExplanationFactor items."""
        items: list[StandardizedEvidence] = []
        for exp in explanations:
            factors = getattr(exp, "factors", [])
            explainer_name = getattr(exp, "explainer_name", "xai_explainer")
            for f in factors:
                feature_name = getattr(f, "feature", "unknown_feature")
                importance = getattr(f, "importance", 0.0)
                direction = getattr(f, "direction", "neutral")

                polarity = EvidencePolarity.NEUTRAL
                if str(direction).endswith("phishing"):
                    polarity = EvidencePolarity.SUPPORTS_PHISHING
                elif str(direction).endswith("legitimate"):
                    polarity = EvidencePolarity.SUPPORTS_LEGITIMATE

                finding_desc = f"Feature '{feature_name}' strongly influenced decision (importance={importance:+.3f})"
                items.append(
                    StandardizedEvidence(
                        source=f"xai_{explainer_name}",
                        category=EvidenceCategory.XAI,
                        finding=finding_desc,
                        polarity=polarity,
                        score=float(importance),
                        score_type=ScoreType.IMPORTANCE,
                        confidence=min(1.0, abs(importance)),
                        supporting_explanation=f"XAI attribute '{feature_name}' indicates {polarity.value}.",
                        provenance={"explainer": explainer_name, "feature": feature_name},
                        status=EvidenceStatus.AVAILABLE,
                    )
                )
        return items
