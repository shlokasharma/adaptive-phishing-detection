"""
Common Evidence Schema and Data Structures for Phase 7 (Evidence Synthesis & LLM Reasoning).

Defines a standardized representation for evidence produced by specialized agents,
traditional ML models, URL detectors, rule-based analyzers, and XAI explainers.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class EvidencePolarity(str, Enum):
    """Polarity of an evidence item regarding phishing likelihood."""

    SUPPORTS_PHISHING = "supports_phishing"
    SUPPORTS_LEGITIMATE = "supports_legitimate"
    INCONCLUSIVE = "inconclusive"
    NEUTRAL = "neutral"


class EvidenceCategory(str, Enum):
    """Categorical source domain of evidence."""

    EMAIL = "email"
    URL = "url"
    SENDER = "sender"
    TRADITIONAL_ML = "traditional_ml"
    TRANSFORMER = "transformer"
    XAI = "xai"
    RULE = "rule"
    ORCHESTRATION = "orchestration"
    THREAT_INTELLIGENCE = "threat_intelligence"
    UNKNOWN = "unknown"


class ScoreType(str, Enum):
    """Interpretation of a numeric score attached to evidence."""

    PROBABILITY = "probability"
    CONFIDENCE = "confidence"
    RISK_SCORE = "risk_score"
    IMPORTANCE = "importance"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    """Availability status of the evidence component."""

    AVAILABLE = "available"
    MISSING = "missing"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class RiskLevel(str, Enum):
    """Overall assessment risk level."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class StandardizedEvidence:
    """
    Standardized, self-describing evidence item.

    Captures findings from heterogeneous sources while preserving score semantics,
    provenance, and availability status.
    """

    source: str
    category: EvidenceCategory | str
    finding: str
    polarity: EvidencePolarity | str = EvidencePolarity.NEUTRAL
    prediction: int | None = None
    score: float | None = None
    score_type: ScoreType | str = ScoreType.UNKNOWN
    confidence: float | None = None
    supporting_explanation: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: EvidenceStatus | str = EvidenceStatus.AVAILABLE
    error_message: str | None = None
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.source or not str(self.source).strip():
            raise ValueError("source must not be empty.")

        # Normalize enums
        if not isinstance(self.category, EvidenceCategory):
            try:
                self.category = EvidenceCategory(str(self.category).lower())
            except ValueError:
                self.category = EvidenceCategory.UNKNOWN

        if not isinstance(self.polarity, EvidencePolarity):
            try:
                self.polarity = EvidencePolarity(str(self.polarity).lower())
            except ValueError:
                self.polarity = EvidencePolarity.NEUTRAL

        if not isinstance(self.score_type, ScoreType):
            try:
                self.score_type = ScoreType(str(self.score_type).lower())
            except ValueError:
                self.score_type = ScoreType.UNKNOWN

        if not isinstance(self.status, EvidenceStatus):
            try:
                self.status = EvidenceStatus(str(self.status).lower())
            except ValueError:
                self.status = EvidenceStatus.AVAILABLE

        # Status & finding checks
        if self.status == EvidenceStatus.AVAILABLE:
            if not self.finding or not str(self.finding).strip():
                raise ValueError("finding must not be empty for AVAILABLE evidence.")

        # Validate prediction if present
        if self.prediction is not None:
            if self.prediction not in (0, 1):
                raise ValueError("prediction must be 0, 1, or None.")

        # Validate probability / confidence bounds
        if self.score is not None:
            if not isinstance(self.score, (int, float)):
                raise TypeError("score must be a numeric value or None.")
            if self.score_type == ScoreType.PROBABILITY:
                if not 0.0 <= float(self.score) <= 1.0:
                    raise ValueError(f"probability score must be between 0.0 and 1.0, got {self.score}")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a numeric value or None.")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def is_phishing(self) -> bool:
        return self.polarity == EvidencePolarity.SUPPORTS_PHISHING

    @property
    def is_legitimate(self) -> bool:
        return self.polarity == EvidencePolarity.SUPPORTS_LEGITIMATE

    @property
    def is_available(self) -> bool:
        return self.status == EvidenceStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence item into a JSON-serializable dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "category": self.category.value if isinstance(self.category, EvidenceCategory) else str(self.category),
            "finding": self.finding,
            "polarity": self.polarity.value if isinstance(self.polarity, EvidencePolarity) else str(self.polarity),
            "prediction": self.prediction,
            "score": float(self.score) if self.score is not None else None,
            "score_type": self.score_type.value if isinstance(self.score_type, ScoreType) else str(self.score_type),
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "supporting_explanation": self.supporting_explanation,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "status": self.status.value if isinstance(self.status, EvidenceStatus) else str(self.status),
            "error_message": self.error_message,
        }


@dataclass
class ConflictReport:
    """Detailed report on detected disagreements between components."""

    has_conflict: bool = False
    conflict_type: str = "none"  # "none", "inter_agent_prediction", "score_disagreement", "modality_tension"
    conflicting_sources: list[str] = field(default_factory=list)
    description: str = "No conflicts detected among available evidence."
    resolution_strategy: str = "consensus"
    resolution_rationale: str = "All executed components agree on classification direction."
    confidence_discount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_conflict": self.has_conflict,
            "conflict_type": self.conflict_type,
            "conflicting_sources": list(self.conflicting_sources),
            "description": self.description,
            "resolution_strategy": self.resolution_strategy,
            "resolution_rationale": self.resolution_rationale,
            "confidence_discount": round(float(self.confidence_discount), 4),
        }


@dataclass
class SynthesisAssessment:
    """
    Final synthesized assessment combining evidence, conflict analysis,
    and natural-language reasoning.
    """

    final_verdict: str  # "phishing", "legitimate", "uncertain"
    binary_label: int  # 1 for phishing, 0 for legitimate
    risk_level: RiskLevel | str
    overall_confidence: float
    uncertainty: float
    supporting_phishing_count: int
    supporting_legitimate_count: int
    inconclusive_count: int
    missing_evidence_count: int
    failed_components_count: int
    conflict_report: ConflictReport
    summary: str
    explanation: str
    explanation_source: str = "deterministic_template"  # "deterministic_template", "cloud_llm", "fallback_template"
    evidence_items: list[StandardizedEvidence] = field(default_factory=list)
    missing_components: list[str] = field(default_factory=list)
    failed_components: list[dict[str, Any]] = field(default_factory=list)
    synthesis_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.binary_label not in (0, 1):
            raise ValueError("binary_label must be 0 or 1.")
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError("overall_confidence must be in [0.0, 1.0].")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0.0, 1.0].")
        if not isinstance(self.risk_level, RiskLevel):
            try:
                self.risk_level = RiskLevel(str(self.risk_level).lower())
            except ValueError:
                self.risk_level = RiskLevel.UNCERTAIN

    def to_dict(self) -> dict[str, Any]:
        """Convert synthesized assessment into a JSON-serializable dictionary."""
        return {
            "final_verdict": self.final_verdict,
            "binary_label": self.binary_label,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else str(self.risk_level),
            "overall_confidence": round(float(self.overall_confidence), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "supporting_phishing_count": self.supporting_phishing_count,
            "supporting_legitimate_count": self.supporting_legitimate_count,
            "inconclusive_count": self.inconclusive_count,
            "missing_evidence_count": self.missing_evidence_count,
            "failed_components_count": self.failed_components_count,
            "conflict_report": self.conflict_report.to_dict() if hasattr(self.conflict_report, "to_dict") else self.conflict_report,
            "summary": self.summary,
            "explanation": self.explanation,
            "explanation_source": self.explanation_source,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "missing_components": list(self.missing_components),
            "failed_components": list(self.failed_components),
            "synthesis_metadata": dict(self.synthesis_metadata),
        }
