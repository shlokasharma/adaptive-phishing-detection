# Phase 7: Evidence Synthesis & LLM Reasoning

## Purpose

Phase 7 introduces an **Evidence Synthesis and Grounded Reasoning layer** to the Adaptive Phishing Detection system. While Phases 3, 5, and 6 focus on orchestrating and selecting which specialized agents to run, Phase 7 is responsible for:

1. **Standardizing Evidence**: Normalizing heterogeneous outputs across Email, URL, and Sender agents, ML baselines, and XAI explainers into a canonical schema.
2. **Preserving Provenance**: Tracking which agent, model, feature, and execution step generated each finding.
3. **Detecting & Resolving Conflicts**: Explicitly identifying prediction disagreements, probability spreads, and cross-modality tensions using defensible security rules rather than naive score averaging.
4. **Final Risk Assessment**: Producing a nuanced assessment (`critical`, `high`, `medium`, `low`, `uncertain`) with confidence discount adjustments.
5. **Grounded Natural-Language Explanations**: Generating readable explanations using deterministic Python logic and templates **without requiring a local LLM, GPU, or external paid service**.
6. **Optional Cloud LLM Integration**: Supporting an opt-in, privacy-preserving cloud adapter that is **strictly disabled by default** and includes prompt-injection defenses and deterministic fallback.

---

## Architectural Pipeline

```
User Input (Email / URL / Sender)
     │
     ▼
Phase 3 / 5 / 6 Orchestrator (Adaptive / RL / Fixed)
     │
     ▼
OrchestrationResult
  ├── agent_results (EmailAgent, URLAgent, SenderAgent)
  ├── decision_trace
  └── evidence items
     │
     ▼
Phase 7 EvidenceSynthesisService
  ├── 1. EvidenceCollector: Normalization, deduplication, missing/failed component tracking
  ├── 2. ConflictResolver: Inter-agent prediction & probability tension analysis
  ├── 3. EvidenceSynthesizer: Risk level determination & confidence discounting
  └── 4. DeterministicExplainer: Grounded template-based explanation (Default, no LLM)
       └── [Optional: CloudLLMAdapter (Disabled by Default)]
     │
     ▼
SynthesisAssessment
  ├── final_verdict ("phishing" | "legitimate" | "uncertain")
  ├── binary_label (0 | 1)
  ├── risk_level ("critical" | "high" | "medium" | "low" | "uncertain")
  ├── overall_confidence & uncertainty
  ├── conflict_report (has_conflict, resolution_strategy, discount)
  ├── summary & full explanation markdown
  ├── missing_components & failed_components
  └── standardized evidence items
```

---

## 1. Common Evidence Schema (`schema.py`)

All component findings are standardized into `StandardizedEvidence`:

| Field | Type | Description |
|---|---|---|
| `evidence_id` | `str` | Deterministic/UUID identifier |
| `source` | `str` | Name of producing component (e.g., `email_analysis_agent`, `xai_email_shap`) |
| `category` | `EvidenceCategory` | Domain: `EMAIL`, `URL`, `SENDER`, `TRADITIONAL_ML`, `TRANSFORMER`, `XAI`, `RULE` |
| `finding` | `str` | Human-readable observation |
| `polarity` | `EvidencePolarity` | Direction: `supports_phishing`, `supports_legitimate`, `inconclusive`, `neutral` |
| `prediction` | `int \| None` | Binary prediction (0, 1, or None) |
| `score` | `float \| None` | Raw score value |
| `score_type` | `ScoreType` | Semantic interpretation: `probability`, `confidence`, `importance`, `risk_score` |
| `confidence` | `float \| None` | Bounded confidence \([0.0, 1.0]\) |
| `supporting_explanation`| `str` | Contextual rationale for the finding |
| `provenance` | `dict` | Source component, model name, execution step, runtime latency |
| `status` | `EvidenceStatus` | `available`, `missing`, `failed`, `unsupported` |
| `error_message` | `str \| None` | Exception details if component failed |

---

## 2. Evidence Collection & Normalization (`collector.py`)

The `EvidenceCollector` extracts findings from:
- `AgentResult` objects (predictive outcomes and internal evidence).
- `OrchestrationStep` traces (execution logs, step timings, agent exceptions).
- `UnifiedExplanation` / `ExplanationFactor` items from Phase 4 XAI.

### Deduplication
Overlapping or repeated observations (e.g. identical keyword indicators or redundant feature factors) are deduplicated via content hashing (`source:category:finding`), preventing artificial inflation of evidence counts.

### Missing & Failed Component Auditing
- Tracks missing inputs: e.g. URL agent omitted when no URL exists in message; Sender agent omitted when headers/sender are absent.
- Explicitly flags unconfigured or offline components (e.g. `transformer_detector`).
- Captures runtime errors without silently defaulting to legitimate classifications.

---

## 3. Conflict Detection and Resolution (`conflict_resolver.py`)

Rather than naively averaging probabilities or selecting the highest score, the `ConflictResolver` applies defensible cybersecurity policies:

1. **Direct Categorical Prediction Contradiction**:
   - Condition: One agent predicts phishing (`1`) while another predicts legitimate (`0`).
   - Strategy: `cautious_asymmetric_security`. Because missed phishing (False Negative) has severe operational consequences, contradictions trigger a security alert, discount confidence by \(0.25\), and prevent unverified dismissal of malicious indicators.
2. **Score / Probability Tension**:
   - Condition: Spread between highest and lowest probability \(\ge 0.35\).
   - Strategy: `evidence_corroboration`. Discount confidence proportionally (\(spread \times 0.2\)) and evaluate against aggregate evidence items.
3. **Modality Tension (Compromised Account Pattern)**:
   - Condition: Sender passes SPF/DKIM authentication, but Email body or URL displays phishing indicators.
   - Strategy: `modality_specificity_hierarchy`. Valid SPF/DKIM indicates only domain legitimacy, not message innocence. Content and URL indicators take precedence; confidence is discounted by \(0.10\) to note the tension.

---

## 4. Grounded Natural-Language Explanations (`explainer.py`)

The default explanation mechanism is `DeterministicExplainer`:
- **Lightweight & Fast**: Pure Python logic and templating. Generates explanations in \(< 1\) ms.
- **No LLM, No GPU**: Operates with 100% local determinism; zero API keys or external dependencies.
- **Truthful & Grounded**: Explanations only describe observed evidence items. Never fabricates indicators or invents malicious URLs without evidence.
- **Sanitized Markdown**: Defangs adversarial input and strips control characters, ensuring prompt injection text cannot alter the explanation structure.

Structured explanation sections:
- **Assessment**: Final verdict, risk level, confidence percentage, and uncertainty.
- **Key Observed Findings**: Directional indicators from each active component with provenance.
- **Agreement & Conflict Analysis**: Details on component consensus, detected disagreements, and applied confidence discounts.
- **Uncertainty & Limitations**: Explicit disclosures of missing modalities, offline models, and component failures.

---

## 5. Optional Cloud LLM Adapter (`llm_adapter.py`)

A provider-independent adapter (`CloudLLMAdapter`) is included for optional cloud LLM reasoning (e.g., OpenAI/Anthropic-compatible endpoints):

### Strict Privacy & Operational Rules:
1. **Disabled by Default**: `LLMConfig(enabled=False)`. Having an API key in the environment **never** activates cloud calls without explicit `enabled=True` configuration.
2. **Zero Local LLM**: No Ollama, Llama, or local neural runtime is downloaded or executed.
3. **Data Minimization**: Never transmits raw email bodies, complete URLs, or credentials. Only transmits sanitized, structured finding descriptions and scores.
4. **Prompt Injection Defense**: System prompt enforces strict evidence-only roles; findings are sanitized and defanged. The LLM is never permitted to override the classification.
5. **Deterministic Fallback**: On any network timeout, HTTP error, rate limit, or malformed JSON, the adapter automatically falls back to `DeterministicExplainer` without breaking the pipeline.

---

## 6. Pipeline & API Integration (`service.py` & `phishing_pipeline.py`)

### `EvidenceSynthesisService`
Provides high-level methods:
- `synthesize(orchestration_result, context, xai_explanations) -> SynthesisAssessment`
- `enrich_pipeline_result(pipeline_dict, orchestration_result, ...) -> dict`

### Backward Compatibility
`PhishingDetectionPipeline` supports Phase 7 synthesis via `enable_synthesis=True`:
- All existing `PipelineResult` keys (`label`, `label_name`, `phishing_probability`, `confidence`, `uncertainty`, `agent_results`, `evidence`, `decision_trace`) remain unchanged.
- New Phase 7 fields (`synthesis`, `final_verdict`, `risk_level`, `explanation`, `decision_summary`) are added cleanly.
- `PipelineResult.to_dict()` and `SynthesisAssessment.to_dict()` are 100% JSON-serializable.

---

## 7. Verification and Test Results

A comprehensive test suite was implemented in `tests/test_evidence_synthesis.py`:

```powershell
# Run Phase 7 dedicated test suite
pytest tests/test_evidence_synthesis.py -v
# 20 passed, 0 failed in 9.94s

# Run combined orchestration regression suite (Phases 3, 5, 6, 7)
pytest tests/test_evidence_synthesis.py tests/test_rl_policy.py tests/test_adaptive_evidence_policy.py tests/test_orchestrator.py tests/test_phishing_pipeline.py -q
# 122 passed, 0 failed in 10.29s

# Run full project test suite
pytest -q
# 507 passed, 7 pre-existing failures in prototype frontend files
```

### Test Coverage Highlights:
- Agreement on phishing and legitimate messages.
- Disagreement handling (Traditional ML vs secondary models, URL vs Email).
- Modality tensions (compromised account pattern).
- Missing modalities (no URL, no sender) and component failure handling.
- Score validation (bounds enforcement and type checking).
- Evidence deduplication.
- Cloud LLM default disabled state, network timeout fallback, and malformed JSON fallback.
- Prompt injection defanging.
- End-to-end pipeline integration with Phase 5 adaptive orchestration and Phase 6 RL policy.
- Full JSON serialization.
