"""
In-Memory Bounded TTL Cache for Threat Intelligence Lookups.

Prevents redundant network queries, respects provider rate limits,
and enforces bounded capacity to avoid memory leaks.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any

from phishing_detection.threat_intelligence.models import (
    ThreatIntelResult,
    ThreatStatus,
)


@dataclass
class CacheEntry:
    result: ThreatIntelResult
    expires_at: float


class ThreatIntelligenceCache:
    """
    Thread-safe, bounded LRU cache with TTL expiration.
    """

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600) -> None:
        self.max_size = max(1, max_size)
        self.default_ttl = max(1, default_ttl_seconds)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()

    def _make_key(self, provider: str, indicator_value: str) -> str:
        return f"{provider.lower()}:{indicator_value.strip().lower()}"

    def get(self, provider: str, indicator_value: str) -> ThreatIntelResult | None:
        """Retrieve a cached result if not expired."""
        key = self._make_key(provider, indicator_value)
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            now = time.time()
            if now > entry.expires_at:
                # Expired
                del self._cache[key]
                return None

            # Mark recently used (move to end for LRU)
            self._cache.move_to_end(key)
            cached_copy = entry.result
            cached_copy.cached = True
            return cached_copy

    def set(
        self,
        provider: str,
        indicator_value: str,
        result: ThreatIntelResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store a result in cache. Does NOT cache transient transport errors.
        """
        # Rule: Do not cache transient failures as benign or valid
        if result.status == ThreatStatus.UNAVAILABLE:
            return

        key = self._make_key(provider, indicator_value)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl

        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Evict oldest entry
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(result=result, expires_at=expires_at)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
