"""
End-to-end phishing detection pipeline.

This module provides a high-level interface that connects the
multi-agent orchestration layer with user-facing phishing detection
inputs.

The pipeline intentionally does not implement orchestration logic
itself. Instead, it delegates adaptive agent selection and evidence
aggregation to OrchestratorAgent, and optionally delegates Phase 7
evidence synthesis and reasoning to EvidenceSynthesisService.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from phishing_detection.agents.base import AgentContext
from phishing_detection.orchestration.orchestrator import (
    OrchestrationResult,
    OrchestratorAgent,
)


@dataclass
class PipelineResult:
    """
    Standardized output of the end-to-end phishing detection pipeline.
    """

    label: int
    label_name: str
    phishing_probability: float
    confidence: float
    uncertainty: float

    agent_results: list[Any] = field(default_factory=list)
    decision_trace: list[Any] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)

    pipeline_metadata: dict[str, Any] = field(default_factory=dict)
    synthesis: Any | None = None

    def __post_init__(self) -> None:
        """Validate pipeline result values."""

        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1.")

        if self.label_name not in {"legitimate", "phishing"}:
            raise ValueError(
                "label_name must be 'legitimate' or 'phishing'."
            )

        if not 0.0 <= self.phishing_probability <= 1.0:
            raise ValueError(
                "phishing_probability must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be between 0.0 and 1.0.")

        if self.label == 1 and self.label_name != "phishing":
            raise ValueError(
                "label=1 must correspond to label_name='phishing'."
            )

        if self.label == 0 and self.label_name != "legitimate":
            raise ValueError(
                "label=0 must correspond to label_name='legitimate'."
            )

    @property
    def is_phishing(self) -> bool:
        """Return whether the final decision is phishing."""

        return self.label == 1

    @property
    def is_legitimate(self) -> bool:
        """Return whether the final decision is legitimate."""

        return self.label == 0

    @property
    def agent_count(self) -> int:
        """Return the number of executed agents."""

        return len(self.agent_results)

    @property
    def evidence_count(self) -> int:
        """Return the number of collected evidence items."""

        return len(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete pipeline result to a JSON-compatible
        dictionary.
        """

        serialized_agents = [
            result.to_dict()
            if hasattr(result, "to_dict")
            else result
            for result in self.agent_results
        ]

        serialized_trace = [
            step.to_dict()
            if hasattr(step, "to_dict")
            else step
            for step in self.decision_trace
        ]

        serialized_evidence = [
            item.to_dict()
            if hasattr(item, "to_dict")
            else item
            for item in self.evidence
        ]

        data = {
            "label": self.label,
            "label_name": self.label_name,
            "phishing_probability": self.phishing_probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "agent_results": serialized_agents,
            "decision_trace": serialized_trace,
            "evidence": serialized_evidence,
            "pipeline_metadata": self.pipeline_metadata,
        }

        if self.synthesis is not None:
            data["synthesis"] = (
                self.synthesis.to_dict()
                if hasattr(self.synthesis, "to_dict")
                else self.synthesis
            )
            if hasattr(self.synthesis, "final_verdict"):
                data["final_verdict"] = self.synthesis.final_verdict
            if hasattr(self.synthesis, "risk_level"):
                rl = self.synthesis.risk_level
                data["risk_level"] = rl.value if hasattr(rl, "value") else str(rl)
            if hasattr(self.synthesis, "explanation"):
                data["explanation"] = self.synthesis.explanation
            if hasattr(self.synthesis, "summary"):
                data["decision_summary"] = self.synthesis.summary

        return data


class PhishingDetectionPipeline:
    """
    High-level end-to-end phishing detection pipeline.

    The pipeline creates an AgentContext from user-provided inputs,
    delegates all agent selection and evidence aggregation to the
    OrchestratorAgent, and optionally performs Phase 7 evidence
    synthesis.
    """

    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        pipeline_name: str = "adaptive_phishing_detection_pipeline",
        pipeline_version: str = "0.1.0",
        synthesis_service: Any | None = None,
        enable_synthesis: bool = False,
        threat_intelligence_service: Any | None = None,
        enable_threat_intel: bool = False,
    ) -> None:
        if orchestrator is None:
            raise ValueError("orchestrator must not be None.")

        if not pipeline_name.strip():
            raise ValueError("pipeline_name must not be empty.")

        if not pipeline_version.strip():
            raise ValueError("pipeline_version must not be empty.")

        self.orchestrator = orchestrator
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.synthesis_service = synthesis_service
        self.enable_synthesis = enable_synthesis or (synthesis_service is not None)
        self.threat_intelligence_service = threat_intelligence_service
        self.enable_threat_intel = enable_threat_intel or (threat_intelligence_service is not None)

    def build_context(
        self,
        *,
        email_text: str | None = None,
        url: str | None = None,
        sender: str | None = None,
        headers: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentContext:
        """
        Build an AgentContext from high-level pipeline inputs.
        """

        normalized_email = email_text

        if normalized_email is not None:
            normalized_email = normalized_email.strip()

        normalized_url = url

        if normalized_url is not None:
            normalized_url = normalized_url.strip()

        normalized_sender = sender

        if normalized_sender is not None:
            normalized_sender = normalized_sender.strip()

        context_metadata = dict(metadata or {})

        context_metadata.setdefault(
            "pipeline_name",
            self.pipeline_name,
        )

        context_metadata.setdefault(
            "pipeline_version",
            self.pipeline_version,
        )

        return AgentContext(
            input_text=normalized_email,
            url=normalized_url,
            sender=normalized_sender,
            headers=dict(headers or {}),
            metadata=context_metadata,
        )

    def _create_pipeline_metadata(
        self,
        *,
        context: AgentContext,
        orchestration_result: OrchestrationResult,
        started_at: datetime,
    ) -> dict[str, Any]:
        """
        Create metadata describing the pipeline execution.
        """

        completed_at = datetime.now(timezone.utc)

        execution_time_ms = (
            completed_at - started_at
        ).total_seconds() * 1000.0

        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "execution_timestamp": completed_at.isoformat(),
            "execution_time_ms": round(execution_time_ms, 6),
            "input_modalities": {
                "email": context.input_text is not None,
                "url": context.url is not None,
                "sender": context.sender is not None,
                "headers": bool(context.headers),
            },
            "agent_count": orchestration_result.agent_count,
            "evidence_count": orchestration_result.evidence_count,
            "orchestration_metadata": orchestration_result.metadata,
        }

    def _convert_result(
        self,
        orchestration_result: OrchestrationResult,
        pipeline_metadata: dict[str, Any],
        synthesis_assessment: Any | None = None,
    ) -> PipelineResult:
        """
        Convert an OrchestrationResult into a PipelineResult.
        """

        return PipelineResult(
            label=orchestration_result.label,
            label_name=orchestration_result.label_name,
            phishing_probability=orchestration_result.phishing_probability,
            confidence=orchestration_result.confidence,
            uncertainty=orchestration_result.uncertainty,
            agent_results=orchestration_result.agent_results,
            decision_trace=orchestration_result.decision_trace,
            evidence=orchestration_result.evidence,
            pipeline_metadata=pipeline_metadata,
            synthesis=synthesis_assessment,
        )

    def analyze(
        self,
        *,
        email_text: str | None = None,
        url: str | None = None,
        sender: str | None = None,
        headers: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Execute the complete phishing detection pipeline.

        Parameters
        ----------
        email_text:
            Email body/content to analyze.

        url:
            URL associated with the message.

        sender:
            Sender email address.

        headers:
            Optional email/header metadata.

        metadata:
            Optional additional metadata. URL feature vectors required
            by URLAnalysisAgent can be supplied here.

        Returns
        -------
        PipelineResult
            Complete structured phishing detection result.
        """

        context = self.build_context(
            email_text=email_text,
            url=url,
            sender=sender,
            headers=headers,
            metadata=metadata,
        )

        started_at = datetime.now(timezone.utc)

        orchestration_result = self.orchestrator.analyze(context)

        # Threat Intelligence enrichment (Phase 8)
        ti_batch_results = []
        if self.enable_threat_intel:
            if self.threat_intelligence_service is None:
                from phishing_detection.threat_intelligence.service import (
                    ThreatIntelligenceService,
                )
                self.threat_intelligence_service = ThreatIntelligenceService()

            # Query URL if present
            if context.url:
                ti_batch = self.threat_intelligence_service.query(context.url)
                ti_batch_results.append(ti_batch)
                ti_evidence = self.threat_intelligence_service.to_evidence_list(ti_batch)
                orchestration_result.evidence.extend(ti_evidence)

            # Query sender domain if sender present
            elif context.sender and "@" in context.sender:
                sender_domain = context.sender.split("@")[-1].strip()
                if sender_domain:
                    ti_batch = self.threat_intelligence_service.query(sender_domain)
                    ti_batch_results.append(ti_batch)
                    ti_evidence = self.threat_intelligence_service.to_evidence_list(ti_batch)
                    orchestration_result.evidence.extend(ti_evidence)

        # Synthesis (Phase 7)
        synthesis_assessment = None
        if self.enable_synthesis:
            if self.synthesis_service is None:
                from phishing_detection.synthesis.service import EvidenceSynthesisService
                self.synthesis_service = EvidenceSynthesisService()
            synthesis_assessment = self.synthesis_service.synthesize(
                orchestration_result=orchestration_result,
                context=context,
            )

        pipeline_metadata = self._create_pipeline_metadata(
            context=context,
            orchestration_result=orchestration_result,
            started_at=started_at,
        )

        if ti_batch_results:
            pipeline_metadata["threat_intelligence"] = [b.to_dict() for b in ti_batch_results]

        if synthesis_assessment is not None:
            pipeline_metadata["synthesis"] = synthesis_assessment.to_dict()

        return self._convert_result(
            orchestration_result=orchestration_result,
            pipeline_metadata=pipeline_metadata,
            synthesis_assessment=synthesis_assessment,
        )
