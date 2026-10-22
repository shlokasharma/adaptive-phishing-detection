# Phase 5: Adaptive Evidence Orchestration

## Purpose

Phase 5 extends the deterministic orchestrator introduced in Phase 3 by making
orchestration decisions **evidence-aware**. Instead of evaluating a single agent's
confidence in isolation, the orchestrator now evaluates the *combined* evidence
from all agents that have already executed and decides whether the accumulated
evidence is sufficient to stop or whether another eligible agent should run.

---

## Architecture Overview

```
AgentContext
     │
     ▼
OrchestratorAgent  ←── AdaptiveEvidencePolicy
     │                        │
     │  for each agent:        │
     │    1. budget check      │
     │    2. eligibility       │
     │    3. execute agent     │
     │    4. policy.assess()  ─┘
     │       ├── sufficient → STOP
     │       └── insufficient → continue
     │
     ▼
OrchestrationResult
  ├── label, label_name, phishing_probability
  ├── confidence, uncertainty
  ├── agent_results
  ├── decision_trace
  ├── evidence (combined)
  └── metadata
        ├── policy: "adaptive_evidence_policy"
        ├── confidence_threshold
        ├── min_evidence_count
        ├── max_agents
        ├── agents_executed
        ├── evidence_count
        ├── total_execution_time_ms
        └── policy_assessment  ← EvidencePolicyAssessment.to_dict()
```

---

## Components

### `AdaptiveEvidencePolicy` (`adaptive_policy.py`)

Responsible for evaluating the combined evidence from all agents executed so far.

**Constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.90` | Minimum confidence required to declare evidence sufficient |
| `min_evidence_count` | `1` | Minimum evidence items required across all executed agents |

**`assess(results) → EvidencePolicyAssessment`**

Called after each agent executes with the *full list* of results so far.

### `EvidencePolicyAssessment`

Frozen dataclass returned by `AdaptiveEvidencePolicy.assess()`.

| Field | Type | Description |
|---|---|---|
| `phishing_probability` | `float` | Combined phishing probability |
| `confidence` | `float` | Combined confidence |
| `uncertainty` | `float` | Complement of confidence |
| `evidence_strength` | `float` | Mean evidence strength |
| `evidence_count` | `int` | Total evidence items |
| `agent_count` | `int` | Agents executed so far |
| `sufficient` | `bool` | Whether evidence is sufficient to stop |
| `reason` | `str` | Human-readable explanation |

### `OrchestratorAgent` (`orchestrator.py`)

Extended to accept an optional `policy` parameter. If `None`, a default
`AdaptiveEvidencePolicy` is constructed using `confidence_threshold` and `min_evidence_count=0`
for backward compatibility.

---

## Probability Aggregation

Combined probability is the **arithmetic mean** of the phishing probabilities
from all agent results executed so far:

```
combined_probability = sum(p_i) / n
```

This is the same formula used in `_aggregate_results()` for the final result.
The formula is **transparent, deterministic, and stateless**.

---

## Confidence, Evidence Sufficiency, and Contradiction Handling

After each agent executes, `AdaptiveEvidencePolicy.assess()` is called.

**Evidence is declared sufficient when:**

1. **Confidence requirement:** Combined confidence across executed agents meets `confidence_threshold`, OR the latest agent produced decisive evidence (`confidence >= confidence_threshold`) without contradicting other confident agents.
2. **Evidence count requirement:** Total evidence items across all executed agents meets `min_evidence_count`.
3. **Consensus requirement (no contradiction):** There are no contradictory confident predictions (e.g., one agent predicting phishing with high confidence while another predicts legitimate with high confidence).

Where:
- `combined_confidence = max(combined_probability, 1 - combined_probability)`
- `evidence_count` = total evidence items across all executed agents

**When contradictory predictions are detected**, the policy returns:
- `"Contradictory predictions detected among agents; additional evidence is required to resolve conflict."`
Orchestration continues to the next eligible agent to resolve the ambiguity.

**When evidence count is below minimum**, the policy returns:
- `"The minimum evidence requirement is not yet satisfied."`

**When combined confidence is below threshold**, the policy returns:
- `"Combined confidence is below the threshold; additional available evidence may help."`

**When evidence is sufficient**, the policy returns:
- `"Combined confidence meets the threshold and the minimum evidence requirement is satisfied."`

---

## Execution Budget

The `max_agents` parameter hard-limits the number of agents that execute.
The budget check occurs **before** eligibility and execution:

```
if len(results) >= max_agents:
    record "skip" step → continue to next agent
```

If the budget runs out before the policy is satisfied, orchestration stops at
the budget limit and returns all results collected so far.

---

## Agent Ordering

Agents are attempted in this fixed deterministic order:

1. **Email Analysis Agent** — primary evidence source
2. **URL Analysis Agent** — secondary evidence
3. **Sender Analysis Agent** — tertiary evidence

Each agent's eligibility is verified via `can_analyze(context)` before execution.

---

## Decision Trace

The `OrchestrationResult.decision_trace` is a list of `OrchestrationStep`
objects that explain every orchestration decision.

**Action types:**

| Action | Meaning |
|---|---|
| `execute` | Agent was eligible and selected for execution |
| `stop` | Policy declared evidence sufficient; orchestration halted |
| `continue` | Policy declared evidence insufficient; next agent needed |
| `skip` | Agent was ineligible (no required input) or budget exhausted |
| `error` | Agent raised an exception; result discarded |

**`stop` and `continue` step reasons** contain the policy explanation and
key metrics:

```
Policy: Combined confidence meets the threshold and the minimum evidence
requirement is satisfied. (combined_probability=0.9500, combined_confidence=0.9500,
evidence_count=2)
```

The trace is deterministic. Step numbers are sequential starting from 1.

---

## Agent Exception Handling

If an agent raises an exception during `.analyze()`:

- An `"error"` step is recorded in the decision trace with the exception message.
- The result is **not** added to the results list.
- Orchestration continues to the next eligible agent.
- The policy is **not** assessed after an error (no result to assess).

This prevents a single agent failure from producing a misleading verdict.

---

## Final Result Metadata

The `OrchestrationResult.metadata` dict includes:

```python
{
    "policy": "adaptive_evidence_policy",
    "confidence_threshold": 0.90,
    "min_evidence_count": 1,
    "max_agents": 3,
    "agents_executed": 1,
    "evidence_count": 2,
    "total_execution_time_ms": 12.3,
    "policy_assessment": {
        "phishing_probability": 0.95,
        "confidence": 0.95,
        "uncertainty": 0.05,
        "evidence_strength": 0.72,
        "evidence_count": 2,
        "agent_count": 1,
        "sufficient": True,
        "reason": "Combined confidence meets ..."
    }
}
```

All values in `metadata` are JSON-serializable primitives (no dataclass objects).

---

## Backward Compatibility

All existing `OrchestrationResult` fields and their meanings are preserved:

- `label`, `label_name`, `phishing_probability`, `confidence`, `uncertainty`
- `agent_results`, `decision_trace`, `evidence`, `metadata`
- Properties: `is_phishing`, `is_legitimate`, `agent_count`, `evidence_count`
- `to_dict()` output structure is unchanged

The `OrchestratorAgent` constructor is backward-compatible. All existing
callers that do not pass `policy=` continue to work as before; a default policy
is constructed from the `confidence_threshold` with `min_evidence_count=0`.

---

## Limitations

- **Arithmetic mean** aggregation weights all agents equally. There is no
  learned or expert-weighted fusion of agent probabilities.
- **Fixed agent ordering** (email → URL → sender) is not dynamically adapted
  based on context characteristics.
- **`min_evidence_count`** counts evidence items, not evidence quality; a
  single high-quality evidence item and many redundant items are treated equally.
- This is a deterministic, rule-based policy. It does **not** use reinforcement
  learning, online learning, or any form of model retraining.
- No accuracy improvement claim is made. Actual detection performance depends
  on the underlying agent models.
