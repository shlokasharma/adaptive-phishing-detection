"""
Indicator Normalization, Canonicalization, and SSRF Prevention for Phase 8.

Normalizes URLs, domains, and IP indicators into canonical, safe representations
prior to querying external threat intelligence providers.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from phishing_detection.threat_intelligence.models import (
    IndicatorType,
    ThreatIndicator,
)

MAX_INDICATOR_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}
PRIVATE_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

DOMAIN_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


class IndicatorNormalizer:
    """
    Validates, canonicalizes, and sanitizes input indicators.
    """

    @classmethod
    def normalize(cls, raw_input: str) -> ThreatIndicator:
        """
        Validate and canonicalize a raw URL, domain, or IP string.
        """
        if not raw_input or not isinstance(raw_input, str):
            return ThreatIndicator(
                value="",
                indicator_type=IndicatorType.UNKNOWN,
                raw_input=str(raw_input or ""),
                is_valid=False,
                error_reason="Indicator is empty or not a string.",
            )

        trimmed = raw_input.strip()

        if len(trimmed) > MAX_INDICATOR_LENGTH:
            return ThreatIndicator(
                value=trimmed[:64] + "...",
                indicator_type=IndicatorType.UNKNOWN,
                raw_input=trimmed,
                is_valid=False,
                error_reason=f"Indicator exceeds maximum allowed length ({MAX_INDICATOR_LENGTH}).",
            )

        # Check for dangerous schemes
        lower_input = trimmed.lower()
        if lower_input.startswith(("javascript:", "data:", "file:", "ftp:", "about:")):
            return ThreatIndicator(
                value=trimmed,
                indicator_type=IndicatorType.UNKNOWN,
                raw_input=trimmed,
                is_valid=False,
                error_reason="Unsupported or dangerous URL scheme.",
            )

        # Check if it has a scheme (indicates URL)
        has_scheme = lower_input.startswith(("http://", "https://"))

        if has_scheme:
            return cls._normalize_url(trimmed)

        # Check if it looks like an IP
        is_ip, ip_obj = cls._check_ip(trimmed)
        if is_ip and ip_obj:
            is_private = any(ip_obj in net for net in PRIVATE_IP_NETWORKS)
            if is_private:
                return ThreatIndicator(
                    value=str(ip_obj),
                    indicator_type=IndicatorType.IP,
                    raw_input=trimmed,
                    is_ip=True,
                    is_valid=False,
                    error_reason="Private, loopback, or internal IP address cannot be queried externally.",
                )
            return ThreatIndicator(
                value=str(ip_obj),
                indicator_type=IndicatorType.IP,
                raw_input=trimmed,
                is_ip=True,
                is_valid=True,
            )

        # If it contains a slash, path, or port, it is a schemeless URL
        if "/" in trimmed or ":" in trimmed:
            return cls._normalize_url("http://" + trimmed, original_raw=trimmed)

        # Check if bare domain
        return cls._normalize_domain(trimmed)

    @classmethod
    def _normalize_url(cls, url_str: str, original_raw: str | None = None) -> ThreatIndicator:
        raw_val = original_raw if original_raw is not None else url_str
        try:
            parsed = urlparse(url_str)
        except Exception as exc:
            return ThreatIndicator(
                value=url_str,
                indicator_type=IndicatorType.URL,
                raw_input=raw_val,
                is_valid=False,
                error_reason=f"Failed to parse URL syntax: {exc}",
            )

        scheme = (parsed.scheme or "http").lower()
        if scheme not in ALLOWED_SCHEMES:
            return ThreatIndicator(
                value=url_str,
                indicator_type=IndicatorType.URL,
                raw_input=raw_val,
                is_valid=False,
                error_reason=f"Scheme '{scheme}' is not supported.",
            )

        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return ThreatIndicator(
                value=url_str,
                indicator_type=IndicatorType.URL,
                raw_input=raw_val,
                is_valid=False,
                error_reason="URL does not contain a valid hostname.",
            )

        # Convert IDN/punycode if needed
        try:
            hostname = hostname.encode("idna").decode("ascii")
        except Exception:
            pass  # Keep as-is if idna encoding fails

        # Check for private host / localhost
        if hostname == "localhost":
            return ThreatIndicator(
                value=url_str,
                indicator_type=IndicatorType.URL,
                raw_input=raw_val,
                domain=hostname,
                is_valid=False,
                error_reason="Localhost target cannot be queried externally.",
            )

        is_ip, ip_obj = cls._check_ip(hostname)
        if is_ip and ip_obj:
            if any(ip_obj in net for net in PRIVATE_IP_NETWORKS):
                return ThreatIndicator(
                    value=url_str,
                    indicator_type=IndicatorType.URL,
                    raw_input=raw_val,
                    domain=hostname,
                    is_ip=True,
                    is_valid=False,
                    error_reason="URL references a private or loopback IP address.",
                )

        port = parsed.port
        # Reconstruct canonical URL without default ports
        netloc = hostname
        if port and not (
            (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        ):
            netloc = f"{hostname}:{port}"

        path = parsed.path or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        canonical_url = f"{scheme}://{netloc}{path}{query}"

        return ThreatIndicator(
            value=canonical_url,
            indicator_type=IndicatorType.URL,
            raw_input=raw_val,
            domain=hostname,
            port=port,
            is_ip=is_ip,
            is_valid=True,
        )

    @classmethod
    def _normalize_domain(cls, domain_str: str) -> ThreatIndicator:
        clean = domain_str.strip().lower().rstrip(".")

        # Handle IDN
        try:
            clean = clean.encode("idna").decode("ascii")
        except Exception:
            pass

        if not DOMAIN_REGEX.match(clean):
            return ThreatIndicator(
                value=clean,
                indicator_type=IndicatorType.DOMAIN,
                raw_input=domain_str,
                domain=clean,
                is_valid=False,
                error_reason="Invalid domain name format.",
            )

        return ThreatIndicator(
            value=clean,
            indicator_type=IndicatorType.DOMAIN,
            raw_input=domain_str,
            domain=clean,
            is_valid=True,
        )

    @classmethod
    def _check_ip(cls, text: str) -> tuple[bool, Any | None]:
        try:
            obj = ipaddress.ip_address(text)
            return True, obj
        except ValueError:
            return False, None
