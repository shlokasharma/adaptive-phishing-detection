"""
Phase 6: Comprehensive tests for Reinforcement Learning policy orchestration.

Covers:
1. State representation (dimensions, ordering, sanitization, no label leakage).
2. Action space (mapping, masking, eligibility, budget constraints).
3. Reward function (TP, TN, FP, FN, step cost, premature penalty, single terminal).
4. Policy learning (TabularQPolicy, DeepQPolicy, training updates, seeds).
5. Persistence & metadata (save, load, corruption, fallback).
6. Orchestration integration (RL policy selection, decision trace, exception safety).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.orchestration.adaptive_policy import (
    AdaptiveEvidencePolicy,
    EvidencePolicyAssessment,
)
from phishing_detection.orchestration.orchestrator import (
    OrchestratorAgent,
    OrchestrationResult,
)
from phishing_detection.orchestration.rl_actions import (
    NUM_ACTIONS,
    ACTION_NAMES,
    OrchestrationAction,
    compute_action_mask,
    get_valid_actions,
    get_agent_action_map,
)
from phishing_detection.orchestration.rl_state import (
    STATE_DIMENSION,
    STATE_FEATURE_NAMES,
    RLState,
    encode_state,
)
from phishing_detection.orchestration.rl_reward import (
    RewardConfig,
    calculate_step_reward,
    calculate_terminal_reward,
)
from phishing_detection.orchestration.rl_policy import (
    BaseRLPolicy,
    TabularQPolicy,
    PureNumPyMLP,
    DeepQPolicy,
    RLEvidencePolicy,
)
from phishing_detection.orchestration.rl_environment import (
    PhishingRLEnvironment,
    StepResult,
)


# ============================================================
# MOCK AGENTS & HELPERS
# ============================================================


class MockAgent:
    """Lightweight mock agent for unit testing."""

    def __init__(
        self,
        name: str,
        probability: float = 0.90,
        can_run: bool = True,
        raise_error: bool = False,
    ) -> None:
        self.name = name
        self.probability = probability
        self.can_run = can_run
        self.raise_error = raise_error

    def get_agent_name(self) -> str:
        return self.name

    def can_analyze(self, context: AgentContext) -> bool:
        return self.can_run

    def analyze(self, context: AgentContext) -> AgentResult:
        if self.raise_error:
            raise RuntimeError(f"Simulated error in {self.name}")
        ev = Evidence(
            evidence_type="mock_signal",
            source=self.name,
            description="Mock evidence item",
            polarity="supports_phishing" if self.probability >= 0.5 else "supports_legitimate",
            confidence=max(self.probability, 1.0 - self.probability),
            reliability=0.9,
        )
        return AgentResult.from_probability(
            agent_name=self.name,
            modality="test",
            phishing_probability=self.probability,
            evidence=[ev],
        )


def make_result(
    prob: float,
    agent_name: str = "email_analysis_agent",
    evidence_count: int = 1,
) -> AgentResult:
    evs = [
        Evidence(
            evidence_type="test",
            source=agent_name,
            description="Item",
            polarity="supports_phishing" if prob >= 0.5 else "supports_legitimate",
            confidence=max(prob, 1.0 - prob),
            reliability=0.8,
        )
        for _ in range(evidence_count)
    ]
    return AgentResult.from_probability(
        agent_name=agent_name,
        modality="test",
        phishing_probability=prob,
        evidence=evs,
    )


# ============================================================
# 1. STATE REPRESENTATION TESTS
# ============================================================


def test_state_dimension_and_feature_names():
    """State feature names and dimension match exactly."""
    assert STATE_DIMENSION == 14
    assert len(STATE_FEATURE_NAMES) == 14
    assert "combined_phishing_probability" == STATE_FEATURE_NAMES[0]
    assert "confidence" == STATE_FEATURE_NAMES[1]
    assert "uncertainty" == STATE_FEATURE_NAMES[2]


def test_encode_state_empty_results():
    """Empty results vector initializes safe non-zero defaults."""
    state = encode_state(results=[], available_agents=["email_analysis_agent"])

    assert state.features.shape == (14,)
    # Default combined prob = 0.5, conf = 0.5, unc = 0.5
    assert state.features[0] == pytest.approx(0.5)
    assert state.features[1] == pytest.approx(0.5)
    assert state.features[2] == pytest.approx(0.5)
    # Evidence strength and count are 0
    assert state.features[3] == 0.0
    assert state.features[4] == 0.0
    # Available agent flags
    assert state.features[11] == 1.0  # email available
    assert state.features[12] == 0.0  # url not available


def test_encode_state_with_single_result():
    """Single result encodes correct probabilities and status flags."""
    r = make_result(0.95, agent_name="email_analysis_agent", evidence_count=2)
    state = encode_state(results=[r], available_agents=["email_analysis_agent", "url_analysis_agent"])

    assert state.features[0] == pytest.approx(0.95)
    assert state.features[1] == pytest.approx(0.95)
    assert state.features[2] == pytest.approx(0.05)
    assert state.features[4] == pytest.approx(0.2)  # 2 items / 10.0
    assert state.features[7] == 1.0  # email executed
    assert state.features[8] == 0.0  # url not executed
    assert state.features[10] == 0.0  # no contradiction


def test_encode_state_contradiction_detection():
    """Conflicting confident results set contradiction flag to 1.0."""
    r1 = make_result(0.05, agent_name="email_analysis_agent")  # confident legitimate
    r2 = make_result(0.95, agent_name="url_analysis_agent")    # confident phishing
    state = encode_state(results=[r1, r2], available_agents=[])

    assert state.features[10] == 1.0  # contradiction detected


def test_encode_state_sanitizes_non_finite_values():
    """NaN or infinite inputs are sanitized to valid bounded floats."""
    class BadResult:
        agent_name = "email_analysis_agent"
        phishing_probability = float("nan")
        evidence = []

    state = encode_state(results=[BadResult()], available_agents=[])
    for val in state.features:
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert 0.0 <= val <= 1.0


def test_encode_state_no_ground_truth_leakage():
    """Encode state function signature accepts no target or label argument."""
    import inspect
    sig = inspect.signature(encode_state)
    param_names = list(sig.parameters.keys())
    assert "label" not in param_names
    assert "ground_truth" not in param_names
    assert "target" not in param_names


def test_state_to_dict_and_to_array():
    """RLState exports clean dictionary and array representations."""
    state = encode_state(results=[], available_agents=[])
    d = state.to_dict()
    assert isinstance(d, dict)
    assert len(d) == 14
    assert d["combined_phishing_probability"] == 0.5

    arr = state.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (14,)


# ============================================================
# 2. ACTION SPACE TESTS
# ============================================================


def test_action_enum_and_string_parsing():
    """Actions map to integers and parse from strings correctly."""
    assert OrchestrationAction.RUN_EMAIL_AGENT == 0
    assert OrchestrationAction.RUN_URL_AGENT == 1
    assert OrchestrationAction.RUN_SENDER_AGENT == 2
    assert OrchestrationAction.STOP_AND_CLASSIFY == 3

    assert OrchestrationAction.from_string("RUN_EMAIL_AGENT") == OrchestrationAction.RUN_EMAIL_AGENT
    assert OrchestrationAction.from_string("email") == OrchestrationAction.RUN_EMAIL_AGENT
    assert OrchestrationAction.from_string("STOP") == OrchestrationAction.STOP_AND_CLASSIFY

    with pytest.raises(ValueError):
        OrchestrationAction.from_string("INVALID_ACTION")


def test_action_target_agent_mapping():
    """Target agent names map to correct agent names."""
    assert OrchestrationAction.RUN_EMAIL_AGENT.target_agent_name == "email_analysis_agent"
    assert OrchestrationAction.RUN_URL_AGENT.target_agent_name == "url_analysis_agent"
    assert OrchestrationAction.RUN_SENDER_AGENT.target_agent_name == "sender_analysis_agent"
    assert OrchestrationAction.STOP_AND_CLASSIFY.target_agent_name is None


def test_compute_action_mask_all_available():
    """When all agents available and unexecuted, all actions are valid."""
    mask = compute_action_mask(
        executed_agents=[],
        available_agents=["email_analysis_agent", "url_analysis_agent", "sender_analysis_agent"],
        remaining_budget=3,
    )
    assert mask == [True, True, True, True]


def test_compute_action_mask_prevents_repeated_execution():
    """Executed agent is masked out."""
    mask = compute_action_mask(
        executed_agents=["email_analysis_agent"],
        available_agents=["email_analysis_agent", "url_analysis_agent"],
        remaining_budget=2,
    )
    assert mask[OrchestrationAction.RUN_EMAIL_AGENT] is False
    assert mask[OrchestrationAction.RUN_URL_AGENT] is True
    assert mask[OrchestrationAction.STOP_AND_CLASSIFY] is True


def test_compute_action_mask_zero_budget_leaves_only_stop():
    """When budget is zero, only STOP_AND_CLASSIFY remains valid."""
    mask = compute_action_mask(
        executed_agents=[],
        available_agents=["email_analysis_agent", "url_analysis_agent"],
        remaining_budget=0,
    )
    assert mask == [False, False, False, True]


def test_get_valid_actions():
    """get_valid_actions returns indices of valid actions."""
    valid = get_valid_actions(
        executed_agents=["email_analysis_agent"],
        available_agents=["email_analysis_agent", "url_analysis_agent"],
        remaining_budget=1,
    )
    assert valid == [OrchestrationAction.RUN_URL_AGENT.value, OrchestrationAction.STOP_AND_CLASSIFY.value]


# ============================================================
# 3. REWARD FUNCTION TESTS
# ============================================================


def test_step_reward_valid_and_invalid():
    """Step reward penalizes execution cost or invalid action."""
    config = RewardConfig(step_cost=0.5, invalid_action_penalty=5.0)

    # Valid agent execution step
    r_valid = calculate_step_reward(OrchestrationAction.RUN_EMAIL_AGENT, is_valid=True, config=config)
    assert r_valid == -0.5

    # Invalid action
    r_invalid = calculate_step_reward(OrchestrationAction.RUN_EMAIL_AGENT, is_valid=False, config=config)
    assert r_invalid == -5.0

    # Stop action has 0 intermediate reward (terminal reward computed separately)
    r_stop = calculate_step_reward(OrchestrationAction.STOP_AND_CLASSIFY, is_valid=True, config=config)
    assert r_stop == 0.0


def test_terminal_reward_true_positive():
    """True Positive yields positive reward proportional to confidence."""
    config = RewardConfig(reward_tp=10.0, evidence_strength_weight=0.0)
    result = make_result(0.90)  # pred = 1, conf = 0.90
    reward, breakdown = calculate_terminal_reward([result], ground_truth=1, config=config)

    assert breakdown["outcome"] == "TP"
    assert reward == pytest.approx(10.0 * 0.90)


def test_terminal_reward_true_negative():
    """True Negative yields positive reward proportional to confidence."""
    config = RewardConfig(reward_tn=10.0, evidence_strength_weight=0.0)
    result = make_result(0.10)  # pred = 0, conf = 0.90
    reward, breakdown = calculate_terminal_reward([result], ground_truth=0, config=config)

    assert breakdown["outcome"] == "TN"
    assert reward == pytest.approx(10.0 * 0.90)


def test_terminal_reward_false_negative_penalty_higher_than_fp():
    """False Negative (missed phishing) carries higher penalty than False Positive."""
    config = RewardConfig(penalty_fp=10.0, penalty_fn=25.0, evidence_strength_weight=0.0)

    # False Positive: predicted phishing (0.90), true is legitimate (0)
    r_fp, b_fp = calculate_terminal_reward([make_result(0.90)], ground_truth=0, config=config)
    assert b_fp["outcome"] == "FP"
    assert r_fp == -10.0

    # False Negative: predicted legitimate (0.10), true is phishing (1)
    r_fn, b_fn = calculate_terminal_reward([make_result(0.10)], ground_truth=1, config=config)
    assert b_fn["outcome"] == "FN"
    assert r_fn == -25.0
    assert abs(r_fn) > abs(r_fp)


def test_terminal_reward_premature_stop_penalty():
    """Stopping with zero agents executed incurs premature stop penalty."""
    config = RewardConfig(penalty_premature_stop=5.0)
    reward, breakdown = calculate_terminal_reward([], ground_truth=0, config=config)

    assert breakdown["premature_penalty"] == -5.0


def test_terminal_reward_invalid_ground_truth_raises():
    """Non-binary ground truth raises ValueError."""
    config = RewardConfig()
    with pytest.raises(ValueError):
        calculate_terminal_reward([], ground_truth=2, config=config)


# ============================================================
# 4. TABULAR Q-LEARNING TESTS
# ============================================================


def test_tabular_q_policy_init_and_discretization():
    """TabularQPolicy initializes and discretizes continuous states."""
    policy = TabularQPolicy(learning_rate=0.1, gamma=0.9, epsilon=0.0)
    state = np.array([0.85, 0.85, 0.15, 0.5, 0.2, 0.33, 0.67, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    key = policy.discretize_state(state)
    assert isinstance(key, str)
    assert key.startswith("p")


def test_tabular_q_policy_greedy_selection():
    """Greedy selection chooses action with highest Q-value among valid actions."""
    policy = TabularQPolicy(epsilon=0.0, random_seed=42)
    state = np.zeros(14)
    key = policy.discretize_state(state)

    # Set Q-values: action 1 has 5.0, action 0 has 2.0
    policy.q_table[key] = [2.0, 5.0, 1.0, 0.0]

    # If action 1 is valid, choose it
    a = policy.select_action(state, valid_actions=[0, 1, 3], explore=False)
    assert a == 1

    # If action 1 is masked out, choose action 0
    a_masked = policy.select_action(state, valid_actions=[0, 3], explore=False)
    assert a_masked == 0


def test_tabular_q_policy_update():
    """Q-learning update changes Q-value towards target."""
    policy = TabularQPolicy(learning_rate=0.5, gamma=0.9, epsilon=0.0)
    s1 = np.zeros(14)
    s2 = np.zeros(14)

    # Step update with positive reward
    td_err = policy.update(s1, action=0, reward=10.0, next_state=s2, done=True, valid_next_actions=[])
    key = policy.discretize_state(s1)
    assert policy.q_table[key][0] == 5.0  # 0 + 0.5 * (10 - 0)
    assert td_err == 10.0


def test_tabular_q_policy_save_and_load(tmp_path):
    """TabularQPolicy saves to JSON and loads identical weights."""
    model_file = tmp_path / "tabular_q.json"
    p1 = TabularQPolicy(learning_rate=0.2, gamma=0.95, epsilon=0.05)
    s = np.zeros(14)
    key = p1.discretize_state(s)
    p1.q_table[key] = [1.5, 2.5, 0.5, 3.5]
    p1.save(model_file)

    p2 = TabularQPolicy.load(model_file)
    assert p2.learning_rate == 0.2
    assert p2.q_table[key] == [1.5, 2.5, 0.5, 3.5]


# ============================================================
# 5. DEEP Q-NETWORK (DQN) TESTS
# ============================================================


def test_deep_q_policy_forward_shape():
    """DeepQPolicy forward outputs 4 Q-values."""
    policy = DeepQPolicy(random_seed=42)
    state = np.zeros(14, dtype=np.float32)
    q_vals = policy.get_q_values_for_state(state)
    assert len(q_vals) == 4


def test_deep_q_policy_deterministic_inference():
    """Greedy inference is completely deterministic with fixed seed."""
    policy = DeepQPolicy(random_seed=123)
    state = np.array([0.7, 0.7, 0.3, 0.4, 0.1, 0.33, 0.67, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    a1 = policy.select_action(state, valid_actions=[0, 1, 2, 3], explore=False)
    a2 = policy.select_action(state, valid_actions=[0, 1, 2, 3], explore=False)
    assert a1 == a2


def test_deep_q_policy_learning_update_reduces_td_error():
    """Backprop updates weights and reduces TD error over repeated steps."""
    policy = DeepQPolicy(learning_rate=0.05, gamma=0.95, random_seed=42)
    s = np.ones(14, dtype=np.float32) * 0.5
    next_s = np.ones(14, dtype=np.float32) * 0.6

    initial_err = policy.update(s, action=1, reward=5.0, next_state=next_s, done=False, valid_next_actions=[0, 1, 3])
    for _ in range(15):
        final_err = policy.update(s, action=1, reward=5.0, next_state=next_s, done=False, valid_next_actions=[0, 1, 3])

    assert final_err < initial_err


def test_deep_q_policy_save_and_load(tmp_path):
    """DeepQPolicy weights survive save and load round-trip exactly."""
    model_file = tmp_path / "dqn_model.json"
    p1 = DeepQPolicy(random_seed=42)
    s = np.random.default_rng(42).standard_normal(14).astype(np.float32)
    q_before = p1.get_q_values_for_state(s)

    p1.save(model_file)
    p2 = DeepQPolicy.load(model_file)
    q_after = p2.get_q_values_for_state(s)

    np.testing.assert_allclose(q_before, q_after, rtol=1e-5)


# ============================================================
# 6. RL ENVIRONMENT TESTS
# ============================================================


def test_rl_environment_reset_and_step():
    """Environment step executes agent, updates budget, and transitions state."""
    env = PhishingRLEnvironment(
        email_agent=MockAgent("email_analysis_agent", probability=0.95),
        url_agent=MockAgent("url_analysis_agent", probability=0.85),
        sender_agent=MockAgent("sender_analysis_agent", probability=0.10),
        max_agents=2,
    )
    context = AgentContext(input_text="Sample phishing text")
    state, info = env.reset(context, ground_truth=1)

    assert state.shape == (14,)
    assert info["budget"] == 2

    # Step 1: execute email agent
    res1 = env.step(OrchestrationAction.RUN_EMAIL_AGENT)
    assert res1.done is False
    assert res1.reward < 0.0  # step cost
    assert len(env.results) == 1

    # Step 2: stop and classify
    res2 = env.step(OrchestrationAction.STOP_AND_CLASSIFY)
    assert res2.done is True
    assert res2.reward > 0.0  # TP reward


def test_rl_environment_budget_exhaustion_terminates():
    """Exhausting max_agents budget terminates the episode."""
    env = PhishingRLEnvironment(
        email_agent=MockAgent("email_analysis_agent", probability=0.90),
        url_agent=MockAgent("url_analysis_agent", probability=0.90),
        sender_agent=MockAgent("sender_analysis_agent", probability=0.90),
        max_agents=1,
    )
    context = AgentContext(input_text="Text")
    env.reset(context, ground_truth=1)

    res = env.step(OrchestrationAction.RUN_EMAIL_AGENT)
    assert res.done is True  # Budget of 1 reached
    assert "budget exhausted" in res.info["reason"].lower()


# ============================================================
# 7. RL EVIDENCE POLICY ADAPTER & FALLBACK TESTS
# ============================================================


def test_rl_evidence_policy_fallback_when_file_missing():
    """Missing model file activates fallback to deterministic policy gracefully."""
    rl_policy = RLEvidencePolicy(model_path="non_existent_model_file.json")
    assert rl_policy.is_fallback_active is True

    # select_action uses fallback without crashing
    r = make_result(0.95)
    action, info = rl_policy.select_action(
        results=[r],
        available_agents=["email_analysis_agent", "url_analysis_agent"],
        remaining_budget=2,
    )
    assert info["used_fallback"] is True
    assert isinstance(action, OrchestrationAction)


def test_rl_evidence_policy_action_selection_with_dqn():
    """RLEvidencePolicy wraps DeepQPolicy and selects valid actions."""
    dqn = DeepQPolicy(random_seed=42)
    rl_policy = RLEvidencePolicy(policy=dqn)

    action, info = rl_policy.select_action(
        results=[],
        available_agents=["email_analysis_agent", "url_analysis_agent"],
        remaining_budget=2,
    )
    assert action in (OrchestrationAction.RUN_EMAIL_AGENT, OrchestrationAction.RUN_URL_AGENT, OrchestrationAction.STOP_AND_CLASSIFY)
    assert info["used_fallback"] is False
    assert "q_values" in info


# ============================================================
# 8. ORCHESTRATOR INTEGRATION TESTS (PHASE 6)
# ============================================================


def test_orchestrator_defaults_to_phase5_deterministic():
    """Orchestrator without explicit policy uses deterministic Phase 5 policy."""
    orchestrator = OrchestratorAgent(
        email_agent=MockAgent("email_analysis_agent", probability=0.95),
        url_agent=MockAgent("url_analysis_agent", probability=0.90),
        sender_agent=MockAgent("sender_analysis_agent", probability=0.80),
    )
    context = AgentContext(input_text="Verify account")
    result = orchestrator.analyze(context)

    assert result.metadata["policy"] == "adaptive_evidence_policy"
    assert result.agent_count == 1  # high conf email stops early in Phase 5


def test_orchestrator_integrates_rl_policy():
    """Orchestrator accepts RLEvidencePolicy and executes RL-guided analysis."""
    dqn = DeepQPolicy(random_seed=42)
    rl_policy = RLEvidencePolicy(policy=dqn)

    orchestrator = OrchestratorAgent(
        email_agent=MockAgent("email_analysis_agent", probability=0.95),
        url_agent=MockAgent("url_analysis_agent", probability=0.90),
        sender_agent=MockAgent("sender_analysis_agent", probability=0.80),
        policy=rl_policy,
    )

    context = AgentContext(input_text="Sample email")
    result = orchestrator.analyze(context)

    assert isinstance(result, OrchestrationResult)
    assert result.metadata["policy"] == "rl_evidence_policy"
    assert "rl_policy_metadata" in result.metadata
    assert len(result.decision_trace) >= 1
    # Trace step actions include execute and stop
    actions = [s.action for s in result.decision_trace]
    assert "stop" in actions


def test_orchestrator_rl_enforces_budget_limit():
    """RL policy cannot exceed max_agents even if it chooses repeated actions."""
    dqn = DeepQPolicy(random_seed=42)
    rl_policy = RLEvidencePolicy(policy=dqn)

    orchestrator = OrchestratorAgent(
        email_agent=MockAgent("email_analysis_agent", probability=0.60),
        url_agent=MockAgent("url_analysis_agent", probability=0.60),
        sender_agent=MockAgent("sender_analysis_agent", probability=0.60),
        max_agents=1,
        policy=rl_policy,
    )

    context = AgentContext(input_text="Sample")
    result = orchestrator.analyze(context)
    assert result.agent_count <= 1


def test_orchestrator_rl_handles_agent_exception():
    """Agent error during RL orchestration is caught and logged to trace."""
    # Configure policy to select RUN_EMAIL_AGENT first
    tabular = TabularQPolicy(epsilon=0.0)
    s = np.zeros(14)
    key = tabular.discretize_state(s)
    tabular.q_table[key] = [10.0, 0.0, 0.0, 0.0]
    rl_policy = RLEvidencePolicy(policy=tabular)

    crashing_email = MockAgent("email_analysis_agent", raise_error=True)
    working_url = MockAgent("url_analysis_agent", probability=0.90)

    orchestrator = OrchestratorAgent(
        email_agent=crashing_email,
        url_agent=working_url,
        sender_agent=MockAgent("sender_analysis_agent", probability=0.50),
        policy=rl_policy,
    )

    context = AgentContext(input_text="Sample")
    result = orchestrator.analyze(context)

    assert result.label in (0, 1)
    error_steps = [s for s in result.decision_trace if s.action == "error"]
    assert len(error_steps) >= 1


def test_orchestration_result_to_dict_json_serializable():
    """Final result from RL orchestration serializes cleanly to JSON."""
    dqn = DeepQPolicy(random_seed=42)
    rl_policy = RLEvidencePolicy(policy=dqn)

    orchestrator = OrchestratorAgent(
        email_agent=MockAgent("email_analysis_agent", probability=0.90),
        url_agent=MockAgent("url_analysis_agent", probability=0.90),
        sender_agent=MockAgent("sender_analysis_agent", probability=0.90),
        policy=rl_policy,
    )

    result = orchestrator.analyze(AgentContext(input_text="Sample"))
    d = result.to_dict()

    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert d["metadata"]["policy"] == "rl_evidence_policy"
