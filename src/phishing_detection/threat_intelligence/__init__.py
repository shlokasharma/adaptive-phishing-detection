"""
Phase 8: Threat Intelligence.

Modular threat intelligence layer for normalizing indicators, querying external
reputation providers (VirusTotal, Google Safe Browsing) and local offline reputation,
caching results safely, and providing structured evidence to Phase 7 synthesis.
"""

from __future__ import annotations

from phishing_detection.threat_intelligence.base import (
    BaseThreatIntelligenceProvider,
)
from phishing_detection.threat_intelligence.cache import (
    ThreatIntelligenceCache,
)
from phishing_detection.threat_intelligence.config import (
    ThreatIntelligenceConfig,
)
from phishing_detection.threat_intelligence.models import (
    IndicatorType,
    ThreatIndicator,
    ThreatIntelBatchResult,
    ThreatIntelResult,
    ThreatStatus,
)
from phishing_detection.threat_intelligence.normalizer import (
    IndicatorNormalizer,
)
from phishing_detection.threat_intelligence.providers import (
    GoogleSafeBrowsingProvider,
    LocalReputationProvider,
    VirusTotalProvider,
)
from phishing_detection.threat_intelligence.service import (
    ThreatIntelligenceService,
)

__all__ = [
    # Models & enums
    "IndicatorType",
    "ThreatStatus",
    "ThreatIndicator",
    "ThreatIntelResult",
    "ThreatIntelBatchResult",
    # Normalizer & Cache & Config
    "IndicatorNormalizer",
    "ThreatIntelligenceCache",
    "ThreatIntelligenceConfig",
    # Providers
    "BaseThreatIntelligenceProvider",
    "VirusTotalProvider",
    "GoogleSafeBrowsingProvider",
    "LocalReputationProvider",
    # Service
    "ThreatIntelligenceService",
]
