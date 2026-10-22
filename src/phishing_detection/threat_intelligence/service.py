"""
Threat Intelligence Service for Phase 8.

Coordinates indicator normalization, caching, multi-provider querying,
conflict aggregation, and conversion into standardized pipeline evidence.
"""

from __future__ import annotations

import logging
from typing import Sequence

from phishing_detection.agents.evidence import Evidence
from phishing_detection.threat_intelligence.base import BaseThreatIntelligenceProvider
from phishing_detection.threat_intelligence.cache import ThreatIntelligenceCache
from phishing_detection.threat_intelligence.config import ThreatIntelligenceConfig
from phishing_detection.threat_intelligence.models import (
    IndicatorType,
    ThreatIndicator,
    ThreatIntelBatchResult,
    ThreatIntelResult,
    ThreatStatus,
)
from phishing_detection.threat_intelligence.normalizer import IndicatorNormalizer
from phishing_detection.threat_intelligence.providers.google_safe_browsing import (
    GoogleSafeBrowsingProvider,
)
from phishing_detection.threat_intelligence.providers.local_reputation import (
    LocalReputationProvider,
)
from phishing_detection.threat_intelligence.providers.virustotal import (
    VirusTotalProvider,
)

logger = logging.getLogger(__name__)


class ThreatIntelligenceService:
    """
    Central coordinator for threat intelligence lookups and multi-provider aggregation.
    """

    def __init__(
        self,
        config: ThreatIntelligenceConfig | None = None,
        providers: Sequence[BaseThreatIntelligenceProvider] | None = None,
        cache: ThreatIntelligenceCache | None = None,
    ) -> None:
        self.config = config or ThreatIntelligenceConfig.from_env()
        self.cache = cache or ThreatIntelligenceCache(
            max_size=self.config.cache_max_size,
            default_ttl_seconds=self.config.cache_ttl_seconds,
        )

        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = self._init_default_providers()

    def _init_default_providers(self) -> list[BaseThreatIntelligenceProvider]:
        active_providers: list[BaseThreatIntelligenceProvider] = []

        # Local reputation is offline, safe, and enabled by default
        if self.config.local_reputation_enabled:
            active_providers.append(LocalReputationProvider())

        # External providers are queried ONLY if external threat intel is enabled AND keys exist
        if self.config.enabled:
            if self.config.is_virustotal_configured():
                active_providers.append(
                    VirusTotalProvider(
                        api_key=self.config.virustotal_api_key,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                )

            if self.config.is_google_safe_browsing_configured():
                active_providers.append(
                    GoogleSafeBrowsingProvider(
                        api_key=self.config.google_safe_browsing_api_key,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                )

        return active_providers

    def query(self, raw_indicator: str) -> ThreatIntelBatchResult:
        """
        Normalize an indicator, query eligible providers, and aggregate reputation results.
        """
        indicator = IndicatorNormalizer.normalize(raw_indicator)

        if not indicator.is_valid:
            invalid_res = ThreatIntelResult(
                provider="normalizer",
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.INVALID,
                error_code="INVALID_INDICATOR",
                evidence_description=indicator.error_reason or "Indicator is malformed or invalid.",
            )
            return ThreatIntelBatchResult(
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                results=[invalid_res],
                overall_status=ThreatStatus.INVALID,
                overall_confidence=1.0,
                summary=f"Indicator validation failed: {indicator.error_reason}",
            )

        if not self.providers:
            unavail_res = ThreatIntelResult(
                provider="service",
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.UNAVAILABLE,
                error_code="NO_PROVIDERS_CONFIGURED",
                evidence_description="No threat intelligence providers are enabled or configured.",
            )
            return ThreatIntelBatchResult(
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                results=[unavail_res],
                overall_status=ThreatStatus.UNAVAILABLE,
                overall_confidence=0.5,
                unavailable_count=1,
                summary="Threat intelligence unavailable: no providers configured.",
            )

        provider_results: list[ThreatIntelResult] = []

        for prov in self.providers:
            if not prov.can_handle(indicator):
                continue

            # Check cache
            cached_res = self.cache.get(prov.provider_name, indicator.value)
            if cached_res is not None:
                provider_results.append(cached_res)
                continue

            # Execute provider lookup with safe isolation
            try:
                res = prov.lookup(indicator)
            except Exception as exc:
                logger.error("Provider '%s' raised unexpected exception: %s", prov.provider_name, exc)
                res = ThreatIntelResult(
                    provider=prov.provider_name,
                    indicator=indicator.value,
                    indicator_type=indicator.indicator_type,
                    status=ThreatStatus.UNAVAILABLE,
                    error_code="UNEXPECTED_PROVIDER_ERROR",
                    evidence_description=f"{prov.provider_name} lookup crashed unexpectedly.",
                )

            # Cache valid result
            self.cache.set(prov.provider_name, indicator.value, res)
            provider_results.append(res)

        return self._aggregate_results(indicator, provider_results)

    def _aggregate_results(
        self,
        indicator: ThreatIndicator,
        results: list[ThreatIntelResult],
    ) -> ThreatIntelBatchResult:
        if not results:
            return ThreatIntelBatchResult(
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                overall_status=ThreatStatus.UNKNOWN,
                summary="No eligible providers could process this indicator.",
            )

        malicious_count = sum(1 for r in results if r.status == ThreatStatus.MALICIOUS)
        suspicious_count = sum(1 for r in results if r.status == ThreatStatus.SUSPICIOUS)
        benign_count = sum(1 for r in results if r.status == ThreatStatus.BENIGN)
        unavailable_count = sum(1 for r in results if r.status == ThreatStatus.UNAVAILABLE)

        has_conflict = False
        conflict_desc = "No conflict among providers."

        # Check for provider disagreement
        if malicious_count > 0 and benign_count > 0:
            has_conflict = True
            conflict_desc = (
                f"Provider disagreement: {malicious_count} flagged malicious, "
                f"{benign_count} flagged clean/benign."
            )

        # Compute overall status with asymmetric security priority
        if malicious_count >= 1:
            overall_status = ThreatStatus.MALICIOUS
            # Compute confidence from highest reporting malicious provider
            mal_conf = max([r.confidence for r in results if r.status == ThreatStatus.MALICIOUS and r.confidence] or [0.85])
            overall_confidence = mal_conf
            summary = f"Threat intelligence: Flagged as MALICIOUS by {malicious_count} provider(s)."
        elif suspicious_count >= 1:
            overall_status = ThreatStatus.SUSPICIOUS
            overall_confidence = 0.65
            summary = f"Threat intelligence: Flagged as SUSPICIOUS by {suspicious_count} provider(s)."
        elif benign_count >= 1:
            overall_status = ThreatStatus.BENIGN
            ben_conf = max([r.confidence for r in results if r.status == ThreatStatus.BENIGN and r.confidence] or [0.80])
            overall_confidence = ben_conf
            summary = f"Threat intelligence: Clean reputation reported by {benign_count} provider(s)."
        elif unavailable_count == len(results):
            overall_status = ThreatStatus.UNAVAILABLE
            overall_confidence = 0.50
            summary = "Threat intelligence: All configured providers were unavailable."
        else:
            overall_status = ThreatStatus.UNKNOWN
            overall_confidence = 0.50
            summary = "Threat intelligence: Inconclusive findings."

        return ThreatIntelBatchResult(
            indicator=indicator.value,
            indicator_type=indicator.indicator_type,
            results=results,
            overall_status=overall_status,
            overall_confidence=round(overall_confidence, 4),
            has_conflict=has_conflict,
            conflict_description=conflict_desc,
            malicious_count=malicious_count,
            suspicious_count=suspicious_count,
            benign_count=benign_count,
            unavailable_count=unavailable_count,
            summary=summary,
        )

    def to_evidence_list(self, batch_result: ThreatIntelBatchResult) -> list[Evidence]:
        """
        Convert batch results into Phase 3/7 Evidence items ready for synthesis.
        """
        evidence_items: list[Evidence] = []
        for res in batch_result.results:
            if res.is_available:
                evidence_items.append(res.to_evidence())

        # If overall status is available, also produce a high-level aggregate summary evidence
        if batch_result.overall_status in (ThreatStatus.MALICIOUS, ThreatStatus.SUSPICIOUS, ThreatStatus.BENIGN):
            polarity = "neutral"
            if batch_result.overall_status in (ThreatStatus.MALICIOUS, ThreatStatus.SUSPICIOUS):
                polarity = "supports_phishing"
            elif batch_result.overall_status == ThreatStatus.BENIGN:
                polarity = "supports_legitimate"

            evidence_items.append(
                Evidence(
                    evidence_type="threat_intelligence",
                    source="threat_intel_service",
                    description=batch_result.summary,
                    polarity=polarity,
                    confidence=batch_result.overall_confidence,
                    reliability=0.95,
                    metadata={
                        "indicator": batch_result.indicator,
                        "overall_status": batch_result.overall_status.value,
                        "has_conflict": batch_result.has_conflict,
                    },
                )
            )

        return evidence_items
