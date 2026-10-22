"""
VirusTotal v3 Reputation Provider Adapter for Phase 8.

Implements the official VirusTotal API v3 URL and Domain report lookup.
Documentation: https://developers.virustotal.com/reference/overview
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from phishing_detection.threat_intelligence.base import BaseThreatIntelligenceProvider
from phishing_detection.threat_intelligence.models import (
    IndicatorType,
    ThreatIndicator,
    ThreatIntelResult,
    ThreatStatus,
)

logger = logging.getLogger(__name__)


def _url_to_vt_id(url: str) -> str:
    """Encode URL to VirusTotal base64 identifier without padding."""
    return base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").strip("=")


class VirusTotalProvider(BaseThreatIntelligenceProvider):
    """
    VirusTotal API v3 reputation provider.
    """

    provider_name: str = "VirusTotal"
    supported_indicator_types: set[IndicatorType] = {IndicatorType.URL, IndicatorType.DOMAIN}

    BASE_URL = "https://www.virustotal.com/api/v3"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 4.0,
        http_client: Any | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def lookup(self, indicator: ThreatIndicator) -> ThreatIntelResult:
        """Query VirusTotal for URL or Domain reputation."""
        if not self.is_configured():
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.UNAVAILABLE,
                error_code="NOT_CONFIGURED",
                evidence_description="VirusTotal API key not configured.",
            )

        if not self.can_handle(indicator):
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.INVALID,
                error_code="UNSUPPORTED_INDICATOR",
                evidence_description=f"VirusTotal cannot handle indicator type '{indicator.indicator_type.value}'.",
            )

        # Construct endpoint
        if indicator.indicator_type == IndicatorType.URL:
            url_id = _url_to_vt_id(indicator.value)
            endpoint = f"{self.BASE_URL}/urls/{url_id}"
        elif indicator.indicator_type == IndicatorType.DOMAIN:
            endpoint = f"{self.BASE_URL}/domains/{indicator.value}"
        else:
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.INVALID,
                error_code="UNSUPPORTED_INDICATOR",
            )

        headers = {
            "x-apikey": self.api_key,
            "Accept": "application/json",
            "User-Agent": "AdaptivePhishingDetection/0.1.0",
        }

        try:
            if self.http_client is not None:
                resp_status, resp_data = self.http_client.get(endpoint, headers=headers)
            else:
                import urllib.error
                import urllib.request

                req = urllib.request.Request(endpoint, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_status = resp.status
                    raw_body = resp.read().decode("utf-8")
                    resp_data = json.loads(raw_body)

            return self._parse_response(indicator, resp_status, resp_data)

        except Exception as exc:
            safe_err = self.redact_secrets(str(exc), self.api_key)
            logger.warning("VirusTotal lookup failed for %s: %s", indicator.value, safe_err)

            error_code = "NETWORK_ERROR"
            if "401" in safe_err or "403" in safe_err or "unauthorized" in safe_err.lower():
                error_code = "AUTH_FAILED"
            elif "429" in safe_err or "quota" in safe_err.lower():
                error_code = "RATE_LIMITED"
            elif "404" in safe_err:
                return ThreatIntelResult(
                    provider=self.provider_name,
                    indicator=indicator.value,
                    indicator_type=indicator.indicator_type,
                    status=ThreatStatus.UNKNOWN,
                    confidence=0.50,
                    evidence_description="Indicator has not been analyzed by VirusTotal yet.",
                )
            elif "timed out" in safe_err.lower() or "timeout" in safe_err.lower():
                error_code = "NETWORK_TIMEOUT"

            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.UNAVAILABLE,
                error_code=error_code,
                evidence_description=f"VirusTotal lookup error: {error_code}",
            )

    def _parse_response(
        self,
        indicator: ThreatIndicator,
        status_code: int,
        data: dict[str, Any],
    ) -> ThreatIntelResult:
        if status_code != 200:
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.UNAVAILABLE,
                error_code=f"HTTP_{status_code}",
                evidence_description=f"VirusTotal API returned HTTP {status_code}.",
            )

        attributes = data.get("data", {}).get("attributes", {})
        stats = attributes.get("last_analysis_stats", {})
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        harmless = int(stats.get("harmless", 0))
        undetected = int(stats.get("undetected", 0))
        total = malicious + suspicious + harmless + undetected

        categories = []
        raw_cats = attributes.get("categories", {})
        if isinstance(raw_cats, dict):
            categories = list(set(raw_cats.values()))[:5]

        reputation = attributes.get("reputation", 0)

        # Map VT statistics to ThreatStatus
        if malicious >= 2:
            status = ThreatStatus.MALICIOUS
            confidence = min(0.99, 0.75 + (malicious / max(1, total)) * 0.24)
            desc = (
                f"VirusTotal: {malicious}/{total} security engines flagged indicator as malicious "
                f"(reputation score: {reputation})."
            )
        elif malicious == 1 or suspicious >= 2:
            status = ThreatStatus.SUSPICIOUS
            confidence = 0.65
            desc = (
                f"VirusTotal: {malicious} engine flagged malicious and {suspicious} flagged suspicious out of {total} engines."
            )
        elif harmless > 0 and malicious == 0:
            status = ThreatStatus.BENIGN
            confidence = min(0.90, 0.70 + (harmless / max(1, total)) * 0.20)
            desc = f"VirusTotal: Clean reputation across {harmless} engines (0 malicious detections out of {total})."
        else:
            status = ThreatStatus.UNKNOWN
            confidence = 0.50
            desc = f"VirusTotal: Inconclusive findings ({total} engines scanned, 0 detections)."

        return ThreatIntelResult(
            provider=self.provider_name,
            indicator=indicator.value,
            indicator_type=indicator.indicator_type,
            status=status,
            confidence=round(confidence, 4),
            score=float(malicious),
            score_description=f"engines_flagged_malicious_out_of_{total}",
            categories=categories,
            evidence_description=desc,
            provenance={
                "provider": self.provider_name,
                "reputation": reputation,
                "stats": stats,
            },
        )
