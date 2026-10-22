"""
Google Safe Browsing v4 Reputation Provider Adapter for Phase 8.

Implements the official Google Safe Browsing Lookup API v4 for URL reputation.
Documentation: https://developers.google.com/safe-browsing/v4/lookup-api
"""

from __future__ import annotations

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


class GoogleSafeBrowsingProvider(BaseThreatIntelligenceProvider):
    """
    Google Safe Browsing Lookup API v4 provider.
    """

    provider_name: str = "GoogleSafeBrowsing"
    supported_indicator_types: set[IndicatorType] = {IndicatorType.URL}

    API_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

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
        """Query Google Safe Browsing Lookup API v4."""
        if not self.is_configured():
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.UNAVAILABLE,
                error_code="NOT_CONFIGURED",
                evidence_description="Google Safe Browsing API key not configured.",
            )

        if not self.can_handle(indicator):
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.INVALID,
                error_code="UNSUPPORTED_INDICATOR",
                evidence_description=f"Google Safe Browsing requires a full URL, got '{indicator.indicator_type.value}'.",
            )

        endpoint = f"{self.API_ENDPOINT}?key={self.api_key}"
        payload = {
            "client": {
                "clientId": "adaptive-phishing-detection",
                "clientVersion": "0.1.0",
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": indicator.value}],
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AdaptivePhishingDetection/0.1.0",
        }

        try:
            if self.http_client is not None:
                resp_status, resp_data = self.http_client.post(endpoint, json=payload, headers=headers)
            else:
                import urllib.error
                import urllib.request

                req_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(endpoint, data=req_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_status = resp.status
                    raw_body = resp.read().decode("utf-8")
                    resp_data = json.loads(raw_body)

            return self._parse_response(indicator, resp_status, resp_data)

        except Exception as exc:
            safe_err = self.redact_secrets(str(exc), self.api_key)
            logger.warning("Google Safe Browsing lookup failed for %s: %s", indicator.value, safe_err)

            error_code = "NETWORK_ERROR"
            if "400" in safe_err or "403" in safe_err or "unauthorized" in safe_err.lower():
                error_code = "AUTH_FAILED"
            elif "429" in safe_err:
                error_code = "RATE_LIMITED"
            elif "timeout" in safe_err.lower() or "timed out" in safe_err.lower():
                error_code = "NETWORK_TIMEOUT"

            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.UNAVAILABLE,
                error_code=error_code,
                evidence_description=f"Google Safe Browsing error: {error_code}",
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
                evidence_description=f"Google Safe Browsing returned HTTP {status_code}.",
            )

        matches = data.get("matches", [])
        if matches:
            threat_types = list(set([m.get("threatType", "MALWARE") for m in matches]))
            desc = f"Google Safe Browsing: Flagged as malicious ({', '.join(threat_types)})."
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.MALICIOUS,
                confidence=0.96,
                score=1.0,
                score_description="threat_match_detected",
                categories=threat_types,
                evidence_description=desc,
                provenance={"matches": matches},
            )
        else:
            # Empty response means clean in GSB v4
            desc = "Google Safe Browsing: No known threat matches detected."
            return ThreatIntelResult(
                provider=self.provider_name,
                indicator=indicator.value,
                indicator_type=indicator.indicator_type,
                status=ThreatStatus.BENIGN,
                confidence=0.88,
                score=0.0,
                score_description="no_threat_matches",
                categories=[],
                evidence_description=desc,
                provenance={"matches_count": 0},
            )
