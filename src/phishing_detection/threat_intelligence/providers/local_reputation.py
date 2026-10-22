"""
Local Curated Reputation Provider for Phase 8.

Provides deterministic, offline reputation checks without requiring an external
network connection or third-party API keys.
"""

from __future__ import annotations

import logging
from typing import Sequence

from phishing_detection.threat_intelligence.base import BaseThreatIntelligenceProvider
from phishing_detection.threat_intelligence.models import (
    IndicatorType,
    ThreatIndicator,
    ThreatIntelResult,
    ThreatStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_KNOWN_MALICIOUS_DOMAINS = {
    "phishing.test",
    "evil-bank.com",
    "account-verification-alert.com",
    "secure-login-update.info",
    "paypal-security-center.net",
    "apple-id-verify.support",
    "microsoft-account-suspended.org",
}

DEFAULT_KNOWN_BENIGN_DOMAINS = {
    "google.com",
    "microsoft.com",
    "apple.com",
    "github.com",
    "paypal.com",
    "amazon.com",
    "netflix.com",
    "wikipedia.org",
    "python.org",
}


class LocalReputationProvider(BaseThreatIntelligenceProvider):
    """
    Offline local threat intelligence provider based on curated blocklists and allowlists.
    """

    provider_name: str = "LocalReputation"
    supported_indicator_types: set[IndicatorType] = {
        IndicatorType.URL,
        IndicatorType.DOMAIN,
        IndicatorType.IP,
    }

    def __init__(
        self,
        malicious_domains: Sequence[str] | None = None,
        benign_domains: Sequence[str] | None = None,
    ) -> None:
        self.malicious_domains = set(
            d.lower() for d in (malicious_domains or DEFAULT_KNOWN_MALICIOUS_DOMAINS)
        )
        self.benign_domains = set(
            d.lower() for d in (benign_domains or DEFAULT_KNOWN_BENIGN_DOMAINS)
        )

    def is_configured(self) -> bool:
        return True  # Always configured and available

    def lookup(self, indicator: ThreatIndicator) -> ThreatIntelResult:
        if not self.can_handle(indicator):
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.INVALID,
                error_code="UNSUPPORTED_INDICATOR",
                evidence_description="LocalReputation cannot handle invalid indicator.",
            )

        domain = indicator.domain or indicator.value
        domain = domain.lower()

        # Check exact domain match
        if domain in self.malicious_domains or any(
            domain.endswith("." + m) for m in self.malicious_domains
        ):
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.MALICIOUS,
                confidence=0.98,
                score=1.0,
                score_description="curated_blocklist_match",
                categories=["known_phishing_domain"],
                evidence_description=f"Local reputation match: '{domain}' is listed on known phishing blocklist.",
                provenance={"source": "local_threat_intelligence_blocklist"},
            )

        if domain in self.benign_domains or any(
            domain.endswith("." + b) for b in self.benign_domains
        ):
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.BENIGN,
                confidence=0.92,
                score=0.0,
                score_description="curated_allowlist_match",
                categories=["trusted_enterprise_domain"],
                evidence_description=f"Local reputation match: '{domain}' is recognized as a trusted enterprise domain.",
                provenance={"source": "local_threat_intelligence_allowlist"},
            )

        # Inconclusive
        return ThreatIntelResult(
            provider=self.provider_name,
            indicator=indicator.value,
            indicator_type=indicator.indicator_type,
            status=ThreatStatus.UNKNOWN,
            confidence=0.50,
            score=None,
            score_description="not_in_local_database",
            categories=[],
            evidence_description=f"Local reputation: '{domain}' is unlisted in local database.",
            provenance={"source": "local_threat_intelligence_database"},
        )
