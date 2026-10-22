# Phase 6: Reinforcement Learning Policy for Adaptive Evidence Orchestration

## Purpose

Phase 6 extends the adaptive orchestration capabilities developed in Phase 5 by introducing a **trainable Reinforcement Learning (RL) policy**. While Phase 5 uses a fixed, rules-based deterministic evidence threshold (`AdaptiveEvidencePolicy`), Phase 6 frames evidence orchestration as a Markov Decision Process (MDP) where a policy agent learns how to allocate analysis budget.

The RL policy learns to dynamically trade off:
- **Detection Accuracy**: Correctly classifying phishing vs. legitimate communications.
- **Asymmetric Security Costs**: Heavy penalties for false negatives (missed phishing attacks) versus false positives.
- **Computational Efficiency**: Minimizing agent executions, reducing execution latency, and adhering to strict budgets.

The implementation is deterministic, bounded, explainable, testable, fully compatible with existing Phase 3–5 pipelines, and runs in pure NumPy with zero heavyweight binary dependencies.

---

## Architecture Overview

```
AgentContext
     │
     ▼
OrchestratorAgent  ←── RLEvidencePolicy (DQN or Tabular Q)
     │                        │
     │  while not terminated:  │
     │    1. state = encode_state(results, context, budget) [14 features]
     │    2. mask = compute_action_mask(context, results, remaining_budget)
     │    3. action = policy.select_action(state, mask)
     │    4. dispatch action:
     │       ├── STOP_AND_CLASSIFY → break loop & aggregate predictions
     │       └── RUN_<AGENT>       → execute agent, record result & step trace
     │    5. if budget exhausted or no valid actions → break loop
     │
     ▼
OrchestrationResult
  ├── label, label_name, phishing_probability
  ├── confidence, uncertainty
  ├── agent_results
  ├── decision_trace (step_number, agent_name, action, reason, confidence)
  ├── evidence (combined evidence from all executed agents)
  └── metadata
        ├── policy: "rl_evidence_policy"
        ├── algorithm: "dqn" | "tabular"
        ├── budget_limit: int
        ├── agents_executed: list[str]
        ├── rl_steps_executed: int
        └── policy_assessment: dict
```

---

## MDP Formulation

### 1. State Representation (`rl_state.py`)

The state is a 14-dimensional continuous feature vector normalized strictly to \([0, 1]\). All features are computed exclusively from the current `AgentContext` and completed `AgentResult` items, guaranteeing **zero ground-truth leakage**:

| Index | Feature Name | Range | Description |
|---|---|---|---|
| 0 | `email_agent_executed` | \(\{0, 1\}\) | Binary indicator if Email Agent has completed |
| 1 | `url_agent_executed` | \(\{0, 1\}\) | Binary indicator if URL Agent has completed |
| 2 | `sender_agent_executed` | \(\{0, 1\}\) | Binary indicator if Sender Agent has completed |
| 3 | `budget_remaining_ratio` | \([0, 1]\) | Ratio of remaining allowed executions to `max_agents` |
| 4 | `context_has_url` | \(\{0, 1\}\) | Whether input context provides a URL |
| 5 | `context_has_sender` | \(\{0, 1\}\) | Whether input context provides sender/headers |
| 6 | `evidence_count_normalized` | \([0, 1]\) | Evidence count clipped at 10 and scaled by 0.1 |
| 7 | `mean_confidence` | \([0, 1]\) | Average confidence across completed agents |
| 8 | `min_confidence` | \([0, 1]\) | Minimum confidence observed |
| 9 | `max_confidence` | \([0, 1]\) | Maximum confidence observed |
| 10 | `phishing_prob_estimate` | \([0, 1]\) | Mean probability across completed agents (0.5 if none) |
| 11 | `prediction_disagreement` | \(\{0, 1\}\) | 1.0 if completed agents disagree on label, else 0.0 |
| 12 | `mean_uncertainty` | \([0, 1]\) | Average uncertainty across completed agents |
| 13 | `mean_evidence_strength` | \([0, 1]\) | Average strength of accumulated evidence items |

All non-finite values (NaN/inf) are automatically sanitized to safe defaults.

### 2. Action Space & Action Masking (`rl_actions.py`)

A discrete 4-action space is defined:
- `0: RUN_EMAIL_AGENT`
- `1: RUN_URL_AGENT`
- `2: RUN_SENDER_AGENT`
- `3: STOP_AND_CLASSIFY`

**Action Masking Rules**:
To prevent invalid agent executions, illegal state loops, and budget violations:
1. `STOP_AND_CLASSIFY` is **always valid** (guaranteeing that the policy can terminate at any step).
2. If remaining budget \(\le 0\), all agent actions are masked out (`mask = [False, False, False, True]`).
3. An agent action is valid **if and only if**:
   - The agent has not already been executed or attempted.
   - The input context contains the required data modality (e.g., URL Agent requires `context.has_url()`, Sender Agent requires `context.has_sender()` or SPF/DKIM headers).

### 3. Reward Function (`rl_reward.py`)

The reward function enforces asymmetric operational penalties:
- **Step Cost**: \(-0.5\) per agent execution to discourage wasteful computation.
- **Terminal True Positive (TP)**: \(+10.0\) for correctly identifying phishing.
- **Terminal True Negative (TN)**: \(+10.0\) for correctly identifying legitimate messages.
- **Terminal False Positive (FP)**: \(-10.0\) for false alarm overhead.
- **Terminal False Negative (FN)**: \(-25.0\) severe penalty for missing an active phishing attack.
- **Premature Stop Penalty**: \(-5.0\) if stopping with zero evidence collected.
- **Invalid Action Attempt**: \(-1.0\) if selecting an action masked as invalid.

---

## Policy Implementations (`rl_policy.py`)

### 1. Deep Q-Network (`DeepQPolicy`)
- **Network**: Pure NumPy 2-layer perceptron (input 14 \(\to\) hidden 64 \(\to\) output 4).
- **Target Network**: Decoupled target weights updated periodically to stabilize TD learning.
- **Loss & Backprop**: Mean squared Bellman error with analytical ReLU and linear gradient propagation.
- **Replay Buffer**: Cyclic experience replay buffer storing \((s, a, r, s', done)\) transitions.
- **Exploration**: \(\epsilon\)-greedy action selection with linear decay.

### 2. Tabular Q-Learning (`TabularQPolicy`)
- Discretizes continuous state features (confidence, budget, evidence, disagreement) into compact integer bin tuples.
- Standard Bellman update: \(Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]\).

### 3. Production Adapter (`RLEvidencePolicy`)
- Wraps trained models (`DeepQPolicy` or `TabularQPolicy`) for production inference.
- Performs greedy evaluation inference with strict action masking.
- Features **automatic deterministic fallback**: if the model JSON artifact is missing or corrupted, it falls back seamlessly to `AdaptiveEvidencePolicy` without throwing runtime exceptions.

---

## Orchestrator Integration (`orchestrator.py`)

`OrchestratorAgent` automatically recognizes whether a policy is deterministic (Phase 5) or RL-based (Phase 6) via duck typing (`hasattr(self.policy, "select_action")`).

When an RL policy is provided, `OrchestratorAgent._analyze_rl()` executes:
1. Initializes `attempted_agents = set()`.
2. Computes the action mask using `compute_action_mask`.
3. Queries `policy.select_action(state, valid_actions)`.
4. Executes the chosen agent, catching exceptions safely and recording them in `decision_trace`.
5. Updates results, checks termination, and produces standard `OrchestrationResult`.
6. Phase 4 Explainability (XAI) and metadata serialization remain 100% backward-compatible.

---

## Training & Evaluation Scripts

1. **Training Script** (`scripts/train_rl_policy.py`):
   ```powershell
   python scripts/train_rl_policy.py --algorithm dqn --episodes 150 --seed 42
   python scripts/train_rl_policy.py --algorithm tabular --episodes 100 --seed 42
   ```
   Artifacts generated:
   - `experiments/models/rl/dqn_policy.json` (and `.meta.json`)
   - `experiments/models/rl/tabular_policy.json` (and `.meta.json`)

2. **Comparative Evaluation Script** (`scripts/evaluate_rl_policy.py`):
   ```powershell
   python scripts/evaluate_rl_policy.py --samples 100 --seed 42
   ```
   Evaluates Fixed Baseline vs. Phase 5 Deterministic vs. Phase 6 RL Policy on identical held-out validation samples.

---

## Empirical Evaluation Results

Evaluated on 100 randomly sampled held-out messages from `data/splits/email/validation.csv` (seed=42):

| Metric | Fixed Baseline (Full Execution) | Phase 5 Deterministic Policy | Phase 6 RL Policy (DQN) |
|---|---|---|---|
| **Accuracy** | 0.9400 | **0.9900** | 0.9400 |
| **Precision** | 1.0000 | 1.0000 | 1.0000 |
| **Recall** | 0.8947 | **0.9825** | 0.8947 |
| **F1-Score** | 0.9444 | **0.9912** | 0.9444 |
| **False Positive Rate** | 0.0000 | 0.0000 | 0.0000 |
| **False Negative Rate** | 0.1053 | **0.0175** | 0.1053 |
| **Avg Agents Executed** | 1.26 | **1.04** | 1.26 |
| **Avg Latency (ms)** | 9.04 ms | **5.36 ms** | 8.66 ms |
| **Mean Reward** | 7.27 | **9.13** | 7.27 |

### Operational Analysis
1. **Phase 5 Deterministic Policy**:
   Achieved the highest operational efficiency (1.04 agents executed per sample, 5.36 ms latency) by terminating early whenever the Email Agent produced high-confidence evidence (\(\ge 0.90\)), invoking the secondary Sender Agent in only 4 cases.
2. **Phase 6 RL Policy**:
   Trained with a severe false-negative penalty (\(-25.0\)), the learned policy adopts a safety-first posture: it prioritizes collecting all available eligible evidence (100 email + 26 sender executions) before stopping. It achieves identical accuracy to the fixed baseline while demonstrating learned action selection.
3. **Phase 5 Remains Safe Default**:
   Because Phase 5 deterministic policy is highly effective on clear-cut cases, `OrchestratorAgent` defaults to `AdaptiveEvidencePolicy`, while allowing seamless switching to `RLEvidencePolicy` when learned allocation is configured.

---

## Verification & Test Results

A dedicated test suite in `tests/test_rl_policy.py` contains 36 comprehensive tests covering:
- State encoding, normalization, non-finite value sanitization, and ground-truth isolation.
- Discrete action enum parsing, mapping, and action masking.
- Step and terminal reward calculation, TP/TN/FP/FN verification, and premature stopping penalties.
- Tabular Q policy updates, serialization, and deserialization.
- Pure NumPy Deep Q-Network forward pass, TD loss reduction, and weight saving/loading.
- Episodic RL environment initialization, transitions, and budget termination.
- Orchestrator integration, budget enforcement, agent exception resilience, and JSON serialization.

### Test Execution Commands
```powershell
# Run Phase 6 RL policy tests
pytest tests/test_rl_policy.py -v
# 36 passed in 10.33s

# Run combined orchestration regression suite (Phases 3, 5, 6)
pytest tests/test_rl_policy.py tests/test_adaptive_evidence_policy.py tests/test_orchestrator.py tests/test_phishing_pipeline.py -q
# 102 passed in 19.38s
```
