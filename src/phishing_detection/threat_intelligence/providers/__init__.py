"""
Threat Intelligence Provider Implementations.
"""

from phishing_detection.threat_intelligence.providers.google_safe_browsing import (
    GoogleSafeBrowsingProvider,
)
from phishing_detection.threat_intelligence.providers.local_reputation import (
    LocalReputationProvider,
)
from phishing_detection.threat_intelligence.providers.virustotal import (
    VirusTotalProvider,
)

__all__ = [
    "VirusTotalProvider",
    "GoogleSafeBrowsingProvider",
    "LocalReputationProvider",
]
