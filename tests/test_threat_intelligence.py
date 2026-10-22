"""
Comprehensive Unit & Integration Test Suite for Phase 8 (Threat Intelligence).

Tests:
1. URL & Domain normalization and canonicalization.
2. Malformed, oversized, and dangerous scheme rejection.
3. SSRF defense: Private IP and localhost detection.
4. VirusTotal response parsing (Malicious, Suspicious, Benign, Unknown).
5. VirusTotal error handling (Auth 401, Rate limit 429, Timeout) and API key redaction.
6. Google Safe Browsing response parsing (Threat match vs Clean).
7. Google Safe Browsing error handling and API key redaction.
8. Local offline reputation provider (Blocklist, Allowlist, Unlisted).
9. In-memory caching: Cache hits, TTL expiration, avoiding caching transient failures, LRU eviction.
10. Multi-provider aggregation and conflict detection.
11. Partial provider failures (one fails, another succeeds without crashing).
12. Graceful behavior when no providers are configured.
13. Conversion to Phase 3/7 Evidence objects.
14. End-to-end integration with PhishingDetectionPipeline and Phase 7 EvidenceSynthesisService.
15. Full JSON serialization of all threat intelligence models.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from phishing_detection.agents.base import AgentContext, BaseAgent
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.orchestration.adaptive_policy import AdaptiveEvidencePolicy
from phishing_detection.orchestration.orchestrator import OrchestratorAgent
from phishing_detection.pipeline.phishing_pipeline import PhishingDetectionPipeline
from phishing_detection.synthesis.schema import EvidenceCategory, RiskLevel
from phishing_detection.synthesis.service import EvidenceSynthesisService
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
from phishing_detection.threat_intelligence.service import ThreatIntelligenceService


class FakeAgent(BaseAgent):
    """Deterministic agent for integration tests."""

    def __init__(self, name: str, modality: str, probability: float) -> None:
        self.name = name
        self.modality = modality
        self.probability = probability

    def get_agent_name(self) -> str:
        return self.name

    def get_modality(self) -> str:
        return self.modality

    def get_capabilities(self) -> list[str]:
        return [f"{self.modality}_analysis"]

    def can_analyze(self, context: AgentContext) -> bool:
        return True

    def analyze(self, context: AgentContext) -> AgentResult:
        return AgentResult.from_probability(
            agent_name=self.name,
            modality=self.modality,
            phishing_probability=self.probability,
        )


# ============================================================
# 1. Normalization & Canonicalization
# ============================================================


def test_url_normalization():
    # Uppercase scheme/host, trailing slash, default port 80
    raw = "  HTTP://EXAMPLE.COM:80/path/to/page?q=test  "
    ind = IndicatorNormalizer.normalize(raw)

    assert ind.is_valid is True
    assert ind.indicator_type == IndicatorType.URL
    assert ind.domain == "example.com"
    # Port 80 stripped for http
    assert ind.value == "http://example.com/path/to/page?q=test"


def test_domain_normalization():
    raw = "  SUBDOMAIN.EXAMPLE.ORG.  "
    ind = IndicatorNormalizer.normalize(raw)

    assert ind.is_valid is True
    assert ind.indicator_type == IndicatorType.DOMAIN
    assert ind.value == "subdomain.example.org"


def test_schemeless_url_recognized():
    raw = "evil-domain.com/login.php"
    ind = IndicatorNormalizer.normalize(raw)

    assert ind.is_valid is True
    assert ind.indicator_type == IndicatorType.URL
    assert ind.domain == "evil-domain.com"
    assert ind.value.startswith("http://evil-domain.com/login.php")


# ============================================================
# 2. Malformed & Dangerous Input Rejection
# ============================================================


def test_empty_and_oversized_rejection():
    ind_empty = IndicatorNormalizer.normalize("   ")
    assert ind_empty.is_valid is False

    ind_huge = IndicatorNormalizer.normalize("http://example.com/" + "a" * 2500)
    assert ind_huge.is_valid is False
    assert "exceeds maximum allowed length" in ind_huge.error_reason


def test_dangerous_schemes_rejected():
    for scheme in ["javascript:alert(1)", "data:text/html;base64,123", "file:///etc/passwd"]:
        ind = IndicatorNormalizer.normalize(scheme)
        assert ind.is_valid is False
        assert "scheme" in ind.error_reason.lower()


# ============================================================
# 3. SSRF Defense: Private IP & Localhost Rejection
# ============================================================


def test_ssrf_private_ip_and_localhost_rejected():
    for target in [
        "127.0.0.1",
        "http://localhost:8080/admin",
        "http://192.168.1.1/router",
        "10.0.0.5",
        "172.16.0.1",
    ]:
        ind = IndicatorNormalizer.normalize(target)
        assert ind.is_valid is False
        assert any(k in ind.error_reason.lower() for k in ("private", "localhost", "internal"))


# ============================================================
# 4. VirusTotal Response Parsing
# ============================================================


def test_virustotal_malicious_response():
    mock_client = MagicMock()
    mock_payload = {
        "data": {
            "attributes": {
                "last_analysis_stats": {
                    "malicious": 14,
                    "suspicious": 2,
                    "harmless": 20,
                    "undetected": 30,
                },
                "categories": {"Forcepoint": "phishing", "Sophos": "malicious"},
                "reputation": -50,
            }
        }
    }
    mock_client.get.return_value = (200, mock_payload)

    provider = VirusTotalProvider(api_key="mock-vt-key", http_client=mock_client)
    ind = IndicatorNormalizer.normalize("http://phishing-site.test/login")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.MALICIOUS
    assert result.confidence >= 0.75
    assert result.score == 14.0
    assert "phishing" in result.categories
    assert "VirusTotal" in result.evidence_description


def test_virustotal_benign_response():
    mock_client = MagicMock()
    mock_payload = {
        "data": {
            "attributes": {
                "last_analysis_stats": {
                    "malicious": 0,
                    "suspicious": 0,
                    "harmless": 65,
                    "undetected": 5,
                },
                "categories": {},
                "reputation": 100,
            }
        }
    }
    mock_client.get.return_value = (200, mock_payload)

    provider = VirusTotalProvider(api_key="mock-vt-key", http_client=mock_client)
    ind = IndicatorNormalizer.normalize("https://google.com")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.BENIGN
    assert result.confidence >= 0.80
    assert result.score == 0.0


def test_virustotal_unknown_404():
    mock_client = MagicMock()
    mock_client.get.side_effect = RuntimeError("HTTP Error 404: Not Found")

    provider = VirusTotalProvider(api_key="mock-vt-key", http_client=mock_client)
    ind = IndicatorNormalizer.normalize("http://newly-seen-domain.xyz")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.UNKNOWN
    assert result.confidence == 0.50


# ============================================================
# 5. VirusTotal Error Handling & Key Redaction
# ============================================================


def test_virustotal_auth_error_and_key_redaction():
    secret_key = "super_secret_vt_key_12345"
    mock_client = MagicMock()
    mock_client.get.side_effect = RuntimeError(f"HTTP Error 401 Unauthorized for key {secret_key}")

    provider = VirusTotalProvider(api_key=secret_key, http_client=mock_client)
    ind = IndicatorNormalizer.normalize("https://example.com")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.UNAVAILABLE
    assert result.error_code == "AUTH_FAILED"
    # Key must not appear in description or error
    assert secret_key not in result.evidence_description


def test_virustotal_rate_limit_error():
    mock_client = MagicMock()
    mock_client.get.side_effect = RuntimeError("HTTP Error 429 Quota Exceeded")

    provider = VirusTotalProvider(api_key="test-key", http_client=mock_client)
    ind = IndicatorNormalizer.normalize("https://example.com")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.UNAVAILABLE
    assert result.error_code == "RATE_LIMITED"


# ============================================================
# 6 & 7. Google Safe Browsing Parsing & Error Handling
# ============================================================


def test_google_safe_browsing_threat_match():
    mock_client = MagicMock()
    mock_payload = {
        "matches": [
            {
                "threatType": "SOCIAL_ENGINEERING",
                "platformType": "ANY_PLATFORM",
                "threat": {"url": "http://evil-banking.test/login"},
            }
        ]
    }
    mock_client.post.return_value = (200, mock_payload)

    provider = GoogleSafeBrowsingProvider(api_key="mock-gsb-key", http_client=mock_client)
    ind = IndicatorNormalizer.normalize("http://evil-banking.test/login")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.MALICIOUS
    assert result.confidence >= 0.90
    assert "SOCIAL_ENGINEERING" in result.categories
    assert "Flagged as malicious" in result.evidence_description


def test_google_safe_browsing_clean_response():
    mock_client = MagicMock()
    mock_client.post.return_value = (200, {})  # Empty dict means clean in GSB v4

    provider = GoogleSafeBrowsingProvider(api_key="mock-gsb-key", http_client=mock_client)
    ind = IndicatorNormalizer.normalize("https://python.org")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.BENIGN
    assert result.confidence >= 0.80
    assert "No known threat matches" in result.evidence_description


def test_google_safe_browsing_key_redaction():
    secret_key = "secret_gsb_key_99999"
    mock_client = MagicMock()
    mock_client.post.side_effect = RuntimeError(f"Connection timeout for {secret_key}")

    provider = GoogleSafeBrowsingProvider(api_key=secret_key, http_client=mock_client)
    ind = IndicatorNormalizer.normalize("https://python.org")
    result = provider.lookup(ind)

    assert result.status == ThreatStatus.UNAVAILABLE
    assert result.error_code == "NETWORK_TIMEOUT"
    assert secret_key not in result.evidence_description


# ============================================================
# 8. Local Reputation Provider (Offline, No Keys)
# ============================================================


def test_local_reputation_blocklist_and_allowlist():
    provider = LocalReputationProvider()

    # Known malicious test domain
    ind_bad = IndicatorNormalizer.normalize("http://phishing.test/login")
    res_bad = provider.lookup(ind_bad)
    assert res_bad.status == ThreatStatus.MALICIOUS
    assert res_bad.confidence >= 0.95

    # Known trusted enterprise domain
    ind_good = IndicatorNormalizer.normalize("https://github.com/login")
    res_good = provider.lookup(ind_good)
    assert res_good.status == ThreatStatus.BENIGN
    assert res_good.confidence >= 0.90

    # Unlisted domain
    ind_unknown = IndicatorNormalizer.normalize("https://unlisted-random-site.test")
    res_unknown = provider.lookup(ind_unknown)
    assert res_unknown.status == ThreatStatus.UNKNOWN


# ============================================================
# 9. In-Memory Bounded TTL Cache
# ============================================================


def test_threat_intelligence_cache_hits_and_expiry():
    cache = ThreatIntelligenceCache(max_size=5, default_ttl_seconds=1)

    res = ThreatIntelResult(
        provider="MockProv",
        indicator="example.com",
        indicator_type=IndicatorType.DOMAIN,
        status=ThreatStatus.BENIGN,
        confidence=0.85,
    )

    # Put in cache
    cache.set("MockProv", "example.com", res, ttl_seconds=1)

    # Cache hit
    cached = cache.get("MockProv", "example.com")
    assert cached is not None
    assert cached.cached is True

    # Wait for TTL to expire
    time.sleep(1.1)
    expired = cache.get("MockProv", "example.com")
    assert expired is None


def test_cache_does_not_store_transient_failures():
    cache = ThreatIntelligenceCache()
    fail_res = ThreatIntelResult(
        provider="MockProv",
        indicator="example.com",
        indicator_type=IndicatorType.DOMAIN,
        status=ThreatStatus.UNAVAILABLE,
        error_code="TIMEOUT",
    )
    cache.set("MockProv", "example.com", fail_res)
    assert cache.get("MockProv", "example.com") is None


def test_cache_lru_eviction():
    cache = ThreatIntelligenceCache(max_size=2)
    for i in range(3):
        res = ThreatIntelResult(
            provider="P",
            indicator=f"site{i}.com",
            indicator_type=IndicatorType.DOMAIN,
            status=ThreatStatus.BENIGN,
        )
        cache.set("P", f"site{i}.com", res)

    # site0 should be evicted, site1 and site2 remain
    assert cache.get("P", "site0.com") is None
    assert cache.get("P", "site1.com") is not None
    assert cache.get("P", "site2.com") is not None


# ============================================================
# 10 & 11. Multi-Provider Aggregation & Conflict Handling
# ============================================================


def test_multi_provider_consensus_and_conflict():
    # Provider A says Malicious, Provider B says Benign
    prov_a = MagicMock(spec=BaseThreatIntelligenceProvider)
    prov_a.provider_name = "ProviderA"
    prov_a.can_handle.return_value = True
    prov_a.lookup.return_value = ThreatIntelResult(
        provider="ProviderA",
        indicator="http://conflict.test",
        indicator_type=IndicatorType.URL,
        status=ThreatStatus.MALICIOUS,
        confidence=0.90,
    )

    prov_b = MagicMock(spec=BaseThreatIntelligenceProvider)
    prov_b.provider_name = "ProviderB"
    prov_b.can_handle.return_value = True
    prov_b.lookup.return_value = ThreatIntelResult(
        provider="ProviderB",
        indicator="http://conflict.test",
        indicator_type=IndicatorType.URL,
        status=ThreatStatus.BENIGN,
        confidence=0.85,
    )

    service = ThreatIntelligenceService(providers=[prov_a, prov_b])
    batch = service.query("http://conflict.test")

    # Conflict explicitly reported
    assert batch.has_conflict is True
    assert "Provider disagreement" in batch.conflict_description
    assert batch.malicious_count == 1
    assert batch.benign_count == 1
    # Asymmetric security priority classifies as MALICIOUS
    assert batch.overall_status == ThreatStatus.MALICIOUS


def test_partial_provider_failure_does_not_crash():
    prov_good = MagicMock(spec=BaseThreatIntelligenceProvider)
    prov_good.provider_name = "GoodProv"
    prov_good.can_handle.return_value = True
    prov_good.lookup.return_value = ThreatIntelResult(
        provider="GoodProv",
        indicator="http://test.test",
        indicator_type=IndicatorType.URL,
        status=ThreatStatus.BENIGN,
    )

    prov_crashing = MagicMock(spec=BaseThreatIntelligenceProvider)
    prov_crashing.provider_name = "CrashingProv"
    prov_crashing.can_handle.return_value = True
    prov_crashing.lookup.side_effect = RuntimeError("Socket reset by peer")

    service = ThreatIntelligenceService(providers=[prov_good, prov_crashing])
    batch = service.query("http://test.test")

    assert batch.overall_status == ThreatStatus.BENIGN
    assert batch.unavailable_count == 1
    assert len(batch.results) == 2


# ============================================================
# 12. No Providers Configured
# ============================================================


def test_no_providers_configured_returns_unavailable():
    config = ThreatIntelligenceConfig(enabled=False, local_reputation_enabled=False)
    service = ThreatIntelligenceService(config=config, providers=[])
    batch = service.query("http://example.com")

    assert batch.overall_status == ThreatStatus.UNAVAILABLE
    assert batch.unavailable_count == 1
    assert "no providers configured" in batch.summary.lower()


# ============================================================
# 13. Conversion to Phase 3/7 Evidence
# ============================================================


def test_conversion_to_evidence():
    res = ThreatIntelResult(
        provider="VirusTotal",
        indicator="http://bad.test",
        indicator_type=IndicatorType.URL,
        status=ThreatStatus.MALICIOUS,
        confidence=0.92,
        categories=["phishing"],
        evidence_description="Flagged as phishing by 12 engines",
    )
    ev = res.to_evidence()

    assert isinstance(ev, Evidence)
    assert ev.evidence_type == "threat_intelligence"
    assert ev.source == "threat_intel_virustotal"
    assert ev.polarity == "supports_phishing"
    assert ev.confidence == 0.92
    assert ev.metadata["threat_status"] == "malicious"


# ============================================================
# 14. End-to-End Integration with Pipeline & Phase 7 Synthesis
# ============================================================


def test_pipeline_with_threat_intelligence_and_synthesis():
    email_agent = FakeAgent("email_analysis_agent", "email", 0.50)  # Inconclusive email
    url_agent = FakeAgent("url_analysis_agent", "url", 0.50)
    sender_agent = FakeAgent("sender_analysis_agent", "sender", 0.50)

    policy = AdaptiveEvidencePolicy(confidence_threshold=0.90)
    orchestrator = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        policy=policy,
        max_agents=3,
    )

    # Use local reputation provider that recognizes phishing.test as malicious
    ti_service = ThreatIntelligenceService(
        config=ThreatIntelligenceConfig(enabled=False, local_reputation_enabled=True)
    )

    pipeline = PhishingDetectionPipeline(
        orchestrator=orchestrator,
        threat_intelligence_service=ti_service,
        enable_threat_intel=True,
        enable_synthesis=True,
    )

    result = pipeline.analyze(
        email_text="Check your account details here.",
        url="http://phishing.test/login",
    )

    # Pipeline executed
    assert result is not None
    # Threat intelligence evidence is injected into evidence list
    ti_ev = [e for e in result.evidence if getattr(e, "evidence_type", "") == "threat_intelligence"]
    assert len(ti_ev) >= 1

    # Metadata records threat intelligence
    assert "threat_intelligence" in result.pipeline_metadata

    # Phase 7 synthesis includes threat intelligence in assessment
    assert result.synthesis is not None
    assert any(
        item.category == EvidenceCategory.THREAT_INTELLIGENCE
        for item in result.synthesis.evidence_items
    )
    # The explanation includes LocalReputation finding
    assert "LocalReputation" in result.synthesis.explanation or "threat_intelligence" in result.synthesis.explanation


# ============================================================
# 15. JSON Serialization
# ============================================================


def test_threat_intelligence_json_serialization():
    res = ThreatIntelResult(
        provider="LocalReputation",
        indicator="http://test.test",
        indicator_type=IndicatorType.URL,
        status=ThreatStatus.MALICIOUS,
        confidence=0.95,
        score=1.0,
        categories=["phishing"],
        evidence_description="Flagged by local blocklist",
    )
    batch = ThreatIntelBatchResult(
        indicator="http://test.test",
        indicator_type=IndicatorType.URL,
        results=[res],
        overall_status=ThreatStatus.MALICIOUS,
        overall_confidence=0.95,
        malicious_count=1,
        summary="Flagged as malicious",
    )

    d = batch.to_dict()
    serialized = json.dumps(d)
    assert len(serialized) > 0

    parsed = json.loads(serialized)
    assert parsed["overall_status"] == "malicious"
    assert len(parsed["provider_results"]) == 1
    assert parsed["provider_results"][0]["provider"] == "LocalReputation"
