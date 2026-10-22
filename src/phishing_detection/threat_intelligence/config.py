"""
Configuration management for Phase 8 (Threat Intelligence).

Supports secure environment variable ingestion, provider enablement,
strict default opt-out for external queries, and credential redaction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _parse_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "on", "enable", "enabled")


@dataclass
class ThreatIntelligenceConfig:
    """
    Threat intelligence system configuration.

    IMPORTANT SAFETY DEFAULTS:
    - `enabled`: Strictly False by default to prevent unexpected external network calls.
    - `local_reputation_enabled`: True by default for zero-network, local offline verification.
    """

    enabled: bool = False
    virustotal_api_key: str | None = None
    google_safe_browsing_api_key: str | None = None
    local_reputation_enabled: bool = True
    timeout_seconds: float = 4.0
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000

    @classmethod
    def from_env(cls) -> "ThreatIntelligenceConfig":
        """Instantiate configuration from environment variables."""
        enabled = _parse_bool(os.getenv("THREAT_INTEL_ENABLED", "false"), default=False)
        vt_key = os.getenv("VIRUSTOTAL_API_KEY", "").strip() or None
        gsb_key = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "").strip() or None
        local_rep = _parse_bool(os.getenv("LOCAL_REPUTATION_ENABLED", "true"), default=True)

        timeout = 4.0
        try:
            timeout = float(os.getenv("THREAT_INTEL_TIMEOUT_SECONDS", "4.0"))
        except ValueError:
            pass

        ttl = 3600
        try:
            ttl = int(os.getenv("THREAT_INTEL_CACHE_TTL_SECONDS", "3600"))
        except ValueError:
            pass

        max_size = 1000
        try:
            max_size = int(os.getenv("THREAT_INTEL_CACHE_MAX_SIZE", "1000"))
        except ValueError:
            pass

        return cls(
            enabled=enabled,
            virustotal_api_key=vt_key,
            google_safe_browsing_api_key=gsb_key,
            local_reputation_enabled=local_rep,
            timeout_seconds=timeout,
            cache_ttl_seconds=ttl,
            cache_max_size=max_size,
        )

    def is_virustotal_configured(self) -> bool:
        return bool(self.virustotal_api_key)

    def is_google_safe_browsing_configured(self) -> bool:
        return bool(self.google_safe_browsing_api_key)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with all credentials safely redacted."""
        return {
            "enabled": self.enabled,
            "virustotal_configured": bool(self.virustotal_api_key),
            "google_safe_browsing_configured": bool(self.google_safe_browsing_api_key),
            "local_reputation_enabled": self.local_reputation_enabled,
            "timeout_seconds": self.timeout_seconds,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_max_size": self.cache_max_size,
        }
