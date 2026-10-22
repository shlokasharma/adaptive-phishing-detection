"""
Base Threat Intelligence Provider interface for Phase 8.

Defines the contract and error handling conventions for all reputation providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from phishing_detection.threat_intelligence.models import (
    IndicatorType,
    ThreatIndicator,
    ThreatIntelResult,
    ThreatStatus,
)

logger = logging.getLogger(__name__)


class BaseThreatIntelligenceProvider(ABC):
    """
    Abstract base class for all threat intelligence reputation providers.
    """

    provider_name: str = "base_provider"
    supported_indicator_types: set[IndicatorType] = {IndicatorType.URL, IndicatorType.DOMAIN}

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required credentials and endpoints are present."""
        raise NotImplementedError

    @abstractmethod
    def lookup(self, indicator: ThreatIndicator) -> ThreatIntelResult:
        """
        Query provider for reputation of a validated indicator.
        """
        raise NotImplementedError

    def can_handle(self, indicator: ThreatIndicator) -> bool:
        """Check if indicator type is supported and indicator is syntactically valid."""
        return indicator.is_valid and indicator.indicator_type in self.supported_indicator_types

    @staticmethod
    def redact_secrets(text: str, secret: str | None) -> str:
        """Redact sensitive API keys or tokens from logs and error strings."""
        if not text or not secret:
            return text or ""
        return str(text).replace(secret, "[REDACTED_API_KEY]")
