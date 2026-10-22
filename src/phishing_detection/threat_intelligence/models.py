"""
Data models and typed schemas for Phase 8 (Threat Intelligence).

Defines standardized indicator types, provider reputation statuses,
individual provider results, and multi-provider aggregated batch results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from phishing_detection.agents.evidence import Evidence


class IndicatorType(str, Enum):
    """Supported threat intelligence indicator types."""

    URL = "url"
    DOMAIN = "domain"
    IP = "ip"
    UNKNOWN = "unknown"


class ThreatStatus(str, Enum):
    """Normalized reputation verdict produced by threat intelligence providers."""

    MALICIOUS = "malicious"
    SUSPICIOUS = "suspicious"
    BENIGN = "benign"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass
class ThreatIndicator:
    """Validated and canonicalized threat indicator."""

    value: str
    indicator_type: IndicatorType
    raw_input: str
    domain: str | None = None
    port: int | None = None
    is_ip: bool = False
    is_valid: bool = True
    error_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "indicator_type": self.indicator_type.value,
            "raw_input": self.raw_input,
            "domain": self.domain,
            "port": self.port,
            "is_ip": self.is_ip,
            "is_valid": self.is_valid,
            "error_reason": self.error_reason,
        }


@dataclass
class ThreatIntelResult:
    """
    Standardized result from an individual threat intelligence provider lookup.
    """

    provider: str
    indicator: str
    indicator_type: IndicatorType
    status: ThreatStatus
    confidence: float | None = None
    score: float | None = None
    score_description: str = ""
    categories: list[str] = field(default_factory=list)
    evidence_description: str = ""
    queried_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cached: bool = False
    error_code: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.indicator_type, IndicatorType):
            try:
                self.indicator_type = IndicatorType(str(self.indicator_type).lower())
            except ValueError:
                self.indicator_type = IndicatorType.UNKNOWN

        if not isinstance(self.status, ThreatStatus):
            try:
                self.status = ThreatStatus(str(self.status).lower())
            except ValueError:
                self.status = ThreatStatus.UNKNOWN

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a number or None.")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0.")

    @property
    def is_malicious(self) -> bool:
        return self.status == ThreatStatus.MALICIOUS

    @property
    def is_suspicious(self) -> bool:
        return self.status == ThreatStatus.SUSPICIOUS

    @property
    def is_benign(self) -> bool:
        return self.status == ThreatStatus.BENIGN

    @property
    def is_available(self) -> bool:
        return self.status not in (ThreatStatus.UNAVAILABLE, ThreatStatus.INVALID)

    def to_evidence(self) -> Evidence:
        """
        Convert this threat intelligence result into a Phase 3/7 Evidence object.
        """
        polarity = "neutral"
        if self.status == ThreatStatus.MALICIOUS:
            polarity = "supports_phishing"
        elif self.status == ThreatStatus.SUSPICIOUS:
            polarity = "supports_phishing"
        elif self.status == ThreatStatus.BENIGN:
            polarity = "supports_legitimate"

        desc = self.evidence_description or f"{self.provider} reported indicator as {self.status.value}"
        meta = {
            "threat_status": self.status.value,
            "provider": self.provider,
            "indicator": self.indicator,
            "indicator_type": self.indicator_type.value,
            "categories": list(self.categories),
            "cached": self.cached,
            "queried_at": self.queried_at,
        }
        if self.error_code:
            meta["error_code"] = self.error_code

        return Evidence(
            evidence_type="threat_intelligence",
            source=f"threat_intel_{self.provider.lower()}",
            description=desc,
            value=self.score if self.score is not None else self.status.value,
            polarity=polarity,
            confidence=self.confidence,
            reliability=0.90 if self.is_available else None,
            metadata=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "provider": self.provider,
            "indicator": self.indicator,
            "indicator_type": self.indicator_type.value,
            "status": self.status.value,
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "score": float(self.score) if self.score is not None else None,
            "score_description": self.score_description,
            "categories": list(self.categories),
            "evidence_description": self.evidence_description,
            "queried_at": self.queried_at,
            "cached": self.cached,
            "error_code": self.error_code,
            "provenance": dict(self.provenance),
        }


@dataclass
class ThreatIntelBatchResult:
    """
    Multi-provider aggregated result for an indicator.
    """

    indicator: str
    indicator_type: IndicatorType
    results: list[ThreatIntelResult] = field(default_factory=list)
    overall_status: ThreatStatus = ThreatStatus.UNKNOWN
    overall_confidence: float = 0.5
    has_conflict: bool = False
    conflict_description: str = "No conflict among providers."
    malicious_count: int = 0
    suspicious_count: int = 0
    benign_count: int = 0
    unavailable_count: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "indicator_type": self.indicator_type.value,
            "overall_status": self.overall_status.value,
            "overall_confidence": round(float(self.overall_confidence), 4),
            "has_conflict": self.has_conflict,
            "conflict_description": self.conflict_description,
            "malicious_count": self.malicious_count,
            "suspicious_count": self.suspicious_count,
            "benign_count": self.benign_count,
            "unavailable_count": self.unavailable_count,
            "summary": self.summary,
            "provider_results": [r.to_dict() for r in self.results],
        }
