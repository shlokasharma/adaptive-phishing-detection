# Phase 8: Threat Intelligence Layer

## Purpose

Phase 8 introduces a **modular Threat Intelligence (TI) reputation layer** into the Adaptive Phishing Detection system. This layer gathers external and local reputation signals for suspicious URLs and domains, normalizes heterogeneous findings into standardized evidence, and seamlessly feeds that evidence into Phase 7 Evidence Synthesis and Explainability.

Key principles:
1. **Zero-Configuration Offline Default**: Includes an offline curated reputation provider (`LocalReputationProvider`) that operates locally without API keys, accounts, or internet access.
2. **Opt-In External Lookups**: External queries are strictly disabled by default (`THREAT_INTEL_ENABLED=false`).
3. **SSRF Prevention & Data Minimization**: Only validated indicators (URLs or domains) are queried. Arbitrary user URLs are never fetched directly by the detection engine. Raw email content, passwords, and cookies are never transmitted.
4. **Resilient Failure Handling**: A provider timeout, rate-limit, or missing API key never crashes the detection pipeline and is never conflated with a clean reputation verdict.
5. **Grounded Provenance**: Evidence flows directly into Phase 7 synthesis, allowing the explainer to cite exact provider findings and represent cross-provider conflicts honestly.

---

## Architectural Flow

```
User Input (Email / URL / Sender)
     │
     ▼
PhishingDetectionPipeline (or OrchestratorAgent)
     │
     ├── 1. Extract Indicator: URL or Sender Domain
     ├── 2. IndicatorNormalizer: Validate syntax, canonicalize host/port, block SSRF/private IPs
     ├── 3. ThreatIntelligenceCache: Check in-memory bounded TTL cache
     │        ├── Cache Hit  → Return cached ThreatIntelResult
     │        └── Cache Miss → Query active providers
     │
     ├── 4. Active Providers:
     │        ├── LocalReputationProvider (offline blocklist / allowlist)
     │        ├── VirusTotalProvider (v3 REST API) [Optional]
     │        └── GoogleSafeBrowsingProvider (v4 Lookup API) [Optional]
     │
     ├── 5. Multi-Provider Aggregator:
     │        ├── Conflict Detection: Flag provider disagreements
     │        └── Asymmetric Security Priority: Prioritize malicious signals
     │
     ├── 6. Convert to Evidence items (Phase 3 & 7 schemas)
     │
     ▼
Phase 7 EvidenceSynthesisService
  ├── 7. Deduplicate & categorize as EvidenceCategory.THREAT_INTELLIGENCE
  ├── 8. Resolve inter-modality tensions (e.g. Email clean vs TI malicious)
  ├── 9. Adjust risk level & compute confidence discounts
  └── 10. Generate grounded natural-language explanation
     │
     ▼
PipelineResult
  ├── label & label_name
  ├── pipeline_metadata["threat_intelligence"]
  └── synthesis (verdict, risk_level, explanation, conflict_report)
```

---

## 1. Components

### `models.py`
- `IndicatorType`: `URL`, `DOMAIN`, `IP`, `UNKNOWN`.
- `ThreatStatus`: `MALICIOUS`, `SUSPICIOUS`, `BENIGN`, `UNKNOWN`, `UNAVAILABLE`, `INVALID`.
- `ThreatIndicator`: Canonicalized indicator with parsed hostname, port, IP flag, and validity status.
- `ThreatIntelResult`: Provider verdict, score, confidence, categories, evidence description, queried timestamp, and provenance. Converts directly to `Evidence` via `to_evidence()`.
- `ThreatIntelBatchResult`: Multi-provider aggregation containing individual results, overall status, conflict descriptions, and provider counts.

### `normalizer.py`
- **IndicatorNormalizer**:
  - Trims whitespace, handles default ports (80/443), and canonicalizes schemes (`http`/`https`).
  - Supports IDN / Punycode domain conversions.
  - Rejects oversized inputs (>2048 chars) and dangerous schemes (`javascript:`, `data:`, `file:`, `ftp:`).
  - **SSRF Prevention**: Inspects destination IPs and blocks private/loopback/internal networks (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `localhost`). External servers are never queried with private addresses.

### `cache.py`
- **ThreatIntelligenceCache**:
  - Thread-safe, bounded LRU cache with configurable TTL (default: 3600 seconds).
  - Bounded size (default: 1000 items) to prevent memory leaks.
  - **Safety Policy**: Never caches transient transport failures (`UNAVAILABLE` network errors) as clean or valid.

### `config.py`
- **ThreatIntelligenceConfig**:
  - Ingests settings from environment variables.
  - Redacts API keys automatically in dictionary and logging representations.

---

## 2. Provider Implementations

| Provider | Type | API Specification | Supported Types | Default Status |
|---|---|---|---|---|
| **`LocalReputationProvider`** | Local / Offline | Curated in-memory blocklists and allowlists | URL, Domain, IP | **Active by default** (zero keys required) |
| **`VirusTotalProvider`** | External REST | VirusTotal v3 API (`/urls/{id}`, `/domains/{domain}`) | URL, Domain | Disabled (requires `VIRUSTOTAL_API_KEY`) |
| **`GoogleSafeBrowsingProvider`** | External REST | Google Safe Browsing Lookup v4 (`threatMatches:find`) | URL | Disabled (requires `GOOGLE_SAFE_BROWSING_API_KEY`) |

### Provider Error Isolation
If one provider crashes or experiences an HTTP timeout, other providers continue executing safely. The pipeline never throws an uncaught exception due to provider downtime.

### API Key Redaction
All provider adapters defang logs and exception strings (`redact_secrets`) ensuring API keys are never leaked to logs, error diagnostics, or frontend payloads.

---

## 3. Configuration & Environment Variables

Create or update a `.env` file from the provided `.env.example`:

| Environment Variable | Default | Description |
|---|---|---|
| `THREAT_INTEL_ENABLED` | `false` | Master switch for external threat intelligence queries |
| `VIRUSTOTAL_API_KEY` | *(None)* | API v3 key for VirusTotal reputation lookups |
| `GOOGLE_SAFE_BROWSING_API_KEY` | *(None)* | API v4 key for Google Safe Browsing lookups |
| `LOCAL_REPUTATION_ENABLED` | `true` | Offline curated blocklist/allowlist verification |
| `THREAT_INTEL_TIMEOUT_SECONDS` | `4.0` | Maximum network timeout per provider request |
| `THREAT_INTEL_CACHE_TTL_SECONDS`| `3600` | In-memory cache time-to-live in seconds |
| `THREAT_INTEL_CACHE_MAX_SIZE` | `1000` | In-memory cache maximum entry capacity |

---

## 4. Multi-Provider Aggregation & Conflict Handling

The `ThreatIntelligenceService` aggregates multiple reputation results:

1. **Consensus**: All active providers agree on malicious, suspicious, or clean.
2. **Provider Disagreement**: E.g., VirusTotal flags 14 malicious engines, but Google Safe Browsing has not yet indexed the URL.
   - Status: `has_conflict = True` with an explicit conflict description.
   - Resolution Policy: Asymmetric security priority. Malicious signals take precedence over absence of detection, avoiding false negatives.
3. **Non-Detection vs. Benign**: Distinguishes unlisted/newly registered domains (`UNKNOWN`) from actively verified benign domains (`BENIGN`).
4. **All Unavailable**: When API keys are unconfigured or networks fail, reports `UNAVAILABLE` rather than defaulting to legitimate.

---

## 5. Pipeline & Phase 7 Synthesis Integration

Threat intelligence integrates into `PhishingDetectionPipeline`:
```python
from phishing_detection.pipeline import PhishingDetectionPipeline
from phishing_detection.threat_intelligence import ThreatIntelligenceService

ti_service = ThreatIntelligenceService()
pipeline = PhishingDetectionPipeline(
    orchestrator=orchestrator,
    threat_intelligence_service=ti_service,
    enable_threat_intel=True,
    enable_synthesis=True,
)

result = pipeline.analyze(
    email_text="Your account is locked. Verify at http://phishing.test/login",
    url="http://phishing.test/login",
)
```

### Integration Evidence Flow:
1. `ThreatIntelligenceService.query(url)` queries providers and generates a `ThreatIntelBatchResult`.
2. `to_evidence_list()` converts findings into Phase 3/7 `Evidence` items.
3. Evidence items are appended to `orchestration_result.evidence`.
4. Phase 7 `EvidenceCollector` maps evidence to `EvidenceCategory.THREAT_INTELLIGENCE`.
5. Phase 7 `DeterministicExplainer` includes threat intelligence findings in the natural-language assessment.
6. `result.pipeline_metadata["threat_intelligence"]` stores complete batch details.

---

## 6. Verification & Test Results

A dedicated test suite in `tests/test_threat_intelligence.py` verifies all functionality:

```powershell
# Run Phase 8 dedicated test suite
pytest tests/test_threat_intelligence.py -v
# 24 passed, 0 failed in 9.91s

# Run combined regression suite (Phases 3, 5, 6, 7, 8)
pytest tests/test_threat_intelligence.py tests/test_evidence_synthesis.py tests/test_rl_policy.py tests/test_adaptive_evidence_policy.py tests/test_orchestrator.py tests/test_phishing_pipeline.py -q
# 146 passed, 0 failed in 10.78s

# Run full project test suite
pytest -q
# 531 passed, 7 pre-existing failures in prototype frontend files
```

### Covered Test Cases:
- URL, domain, and schemeless URL canonicalization.
- Dangerous scheme rejection and length limit enforcement.
- SSRF prevention (private IPs, loopback, localhost).
- VirusTotal malicious, benign, 404 unknown, rate-limit, and timeout responses.
- Google Safe Browsing threat match and clean responses.
- API key redaction in error messages.
- LocalReputationProvider offline blocklists and allowlists.
- Caching hits, TTL eviction, transient error exclusion, and LRU eviction.
- Multi-provider consensus, conflict detection, and partial provider failures.
- Pipeline end-to-end integration and synthesis explanation generation.
- Full JSON serialization.

---

## 7. Known Limitations

1. **Provider Quotas**: Public VirusTotal API keys are limited to 4 requests/minute. The in-memory cache mitigates duplicate requests, but high-throughput pipelines require enterprise keys.
2. **Offline Mode**: When running with default settings (`THREAT_INTEL_ENABLED=false`), external reputation lookups are bypassed and `LocalReputationProvider` provides offline validation.
