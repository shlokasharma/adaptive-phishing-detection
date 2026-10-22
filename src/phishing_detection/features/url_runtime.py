"""
Runtime URL feature construction.

This module converts a raw URL into the feature schema expected by
the frozen UCI Phishing Websites baseline.

Important:
The frozen Phase 2 model was trained on engineered UCI features.
Runtime extraction therefore provides a deterministic deployment
representation rather than downloading or crawling the target page.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse


def _safe_parse(url: str):
    candidate = url.strip()

    if not candidate.startswith(
        ("http://", "https://")
    ):
        candidate = "http://" + candidate

    return urlparse(candidate)


def _domain(url: str) -> str:
    parsed = _safe_parse(url)
    return parsed.netloc.lower().split("@")[-1].split(":")[0]


def _path(url: str) -> str:
    return _safe_parse(url).path or ""


def _is_ip(domain: str) -> int:
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0


def _has_subdomain(domain: str) -> int:
    parts = [
        part
        for part in domain.split(".")
        if part
    ]

    return int(len(parts) > 2)


def _url_length(url: str) -> int:
    return len(url)


def _has_at(url: str) -> int:
    return int("@" in url)


def _has_double_slash_redirect(url: str) -> int:
    scheme_end = url.find("://")

    if scheme_end == -1:
        return 0

    remainder = url[scheme_end + 3:]

    return int("//" in remainder)


def _has_prefix_suffix(domain: str) -> int:
    return int("-" in domain)


def _has_shortening_service(url: str) -> int:
    domain = _domain(url)

    shortening_domains = {
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "rebrand.ly",
        "cutt.ly",
        "shorturl.at",
        "tiny.cc",
    }

    return int(
        any(
            domain == item
            or domain.endswith("." + item)
            for item in shortening_domains
        )
    )


def _https(url: str) -> int:
    return int(
        _safe_parse(url).scheme.lower()
        == "https"
    )


def _has_https_token(url: str) -> int:
    parsed = _safe_parse(url)

    domain = parsed.netloc.lower()

    return int(
        "https" in domain
    )


def _has_port(url: str) -> int:
    parsed = _safe_parse(url)

    try:
        return int(parsed.port is not None)
    except ValueError:
        return 1


def _domain_length(domain: str) -> int:
    return len(domain)


def _digit_count(url: str) -> int:
    return sum(
        character.isdigit()
        for character in url
    )


def _special_character_count(url: str) -> int:
    return len(
        re.findall(
            r"[^a-zA-Z0-9]",
            url,
        )
    )


def _query_length(url: str) -> int:
    return len(
        _safe_parse(url).query
    )


def _path_length(url: str) -> int:
    return len(
        _path(url)
    )


def _subdomain_count(domain: str) -> int:
    parts = [
        part
        for part in domain.split(".")
        if part
    ]

    return max(
        0,
        len(parts) - 2,
    )


def _keyword_count(url: str) -> int:
    suspicious_keywords = (
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "account",
        "update",
        "confirm",
        "password",
        "bank",
        "payment",
        "wallet",
        "invoice",
        "auth",
        "authenticate",
        "webscr",
    )

    lowered = url.lower()

    return sum(
        keyword in lowered
        for keyword in suspicious_keywords
    )


def build_runtime_url_features(
    url: str,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Build a deterministic runtime feature dictionary.

    The returned dictionary follows the exact feature order supplied
    by the frozen preprocessing artifact.
    """

    if not url or not url.strip():
        raise ValueError(
            "URL must not be empty."
        )

    url = url.strip()

    parsed = _safe_parse(url)

    domain = _domain(url)

    path = _path(url)

    features = {
        "having_IP_Address": float(
            _is_ip(domain)
        ),

        "URL_Length": float(
            _url_length(url)
        ),

        "Shortining_Service": float(
            _has_shortening_service(url)
        ),

        "having_At_Symbol": float(
            _has_at(url)
        ),

        "double_slash_redirecting": float(
            _has_double_slash_redirect(url)
        ),

        "Prefix_Suffix": float(
            _has_prefix_suffix(domain)
        ),

        "having_Sub_Domain": float(
            _has_subdomain(domain)
        ),

        "SSLfinal_State": float(
            _https(url)
        ),

        "Domain_registeration_length": 0.0,

        "Favicon": 0.0,

        "port": float(
            _has_port(url)
        ),

        "HTTPS_token": float(
            _has_https_token(url)
        ),

        "Request_URL": 0.0,

        "URL_of_Anchor": 0.0,

        "Links_in_tags": 0.0,

        "SFH": 0.0,

        "Submitting_to_email": float(
            int(
                "mailto:" in url.lower()
            )
        ),

        "Abnormal_URL": float(
            _has_ip_or_unusual_domain(url)
        ),

        "Redirect": 0.0,

        "on_mouseover": 0.0,

        "RightClick": 0.0,

        "popUpWidnow": 0.0,

        "Iframe": 0.0,

        "age_of_domain": 0.0,

        "DNSRecord": 0.0,

        "web_traffic": 0.0,

        "Page_Rank": 0.0,

        "Google_Index": 0.0,

        "Links_pointing_to_page": 0.0,

        "Statistical_report": 0.0,

        # Runtime-only supplementary signals.
        "digit_count": float(
            _digit_count(url)
        ),

        "special_character_count": float(
            _special_character_count(url)
        ),

        "query_length": float(
            _query_length(url)
        ),

        "path_length": float(
            len(path)
        ),

        "subdomain_count": float(
            _subdomain_count(domain)
        ),

        "suspicious_keyword_count": float(
            _keyword_count(url)
        ),
    }

    result = {}

    for feature_name in feature_names:
        if feature_name in features:
            result[feature_name] = features[
                feature_name
            ]
        else:
            result[feature_name] = 0.0

    return result


def _has_ip_or_unusual_domain(
    url: str,
) -> int:
    domain = _domain(url)

    if _is_ip(domain):
        return 1

    return int(
        "." not in domain
        or len(domain) > 100
    )