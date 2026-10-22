"""
Reinforcement Learning Policies for Adaptive Evidence Orchestration (Phase 6).

Implements:
1. BaseRLPolicy: Abstract interface for orchestration policies.
2. TabularQPolicy: Discrete state Q-learning policy.
3. DeepQPolicy: Pure NumPy Deep Q-Network (DQN) with replay buffer and target network.
4. RLEvidencePolicy: Production orchestration adapter with action selection,
   action masking, serialization, and deterministic fallback.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phishing_detection.agents.result import AgentResult
from phishing_detection.orchestration.adaptive_policy import (
    AdaptiveEvidencePolicy,
    EvidencePolicyAssessment,
)
from phishing_detection.orchestration.rl_actions import (
    NUM_ACTIONS,
    OrchestrationAction,
    compute_action_mask,
    get_valid_actions,
)
from phishing_detection.orchestration.rl_state import (
    STATE_DIMENSION,
    RLState,
    encode_state,
)

logger = logging.getLogger(__name__)


# ============================================================
# BASE RL POLICY
# ============================================================


class BaseRLPolicy(ABC):
    """Abstract base class for all RL orchestration policies."""

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        explore: bool = False,
    ) -> int:
        """Select an action index among valid_actions."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_next_actions: Sequence[int],
    ) -> float:
        """Perform a policy update step and return loss/TD error."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model weights and metadata."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseRLPolicy:
        """Load model from file."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return policy metadata and hyperparameters."""
        raise NotImplementedError


# ============================================================
# TABULAR Q-LEARNING POLICY
# ============================================================


class TabularQPolicy(BaseRLPolicy):
    """
    Tabular Q-learning policy with discrete state discretization.

    Parameters
    ----------
    learning_rate:
        Step size alpha for Q-value updates.
    gamma:
        Discount factor for future rewards.
    epsilon:
        Exploration rate for epsilon-greedy selection.
    random_seed:
        Seed for reproducibility.
    """

    POLICY_TYPE = "tabular_q"
    VERSION = "1.0.0"

    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        random_seed: int | None = 42,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.q_table: dict[str, list[float]] = {}

    @staticmethod
    def discretize_state(state: np.ndarray) -> str:
        """
        Discretize continuous state vector into a string key.

        State format:
        [prob, conf, unc, ev_str, ev_count, exec_count, rem_budget,
         email_done, url_done, sender_done, contradiction,
         email_avail, url_avail, sender_avail]
        """
        prob = float(state[0])
        conf = float(state[1])
        rem_budget = float(state[6])
        email_done = int(state[7] >= 0.5)
        url_done = int(state[8] >= 0.5)
        sender_done = int(state[9] >= 0.5)
        contradiction = int(state[10] >= 0.5)

        # Discretize probability: 5 bins
        if prob < 0.2:
            p_bin = 0
        elif prob < 0.4:
            p_bin = 1
        elif prob < 0.6:
            p_bin = 2
        elif prob < 0.8:
            p_bin = 3
        else:
            p_bin = 4

        # Discretize confidence: 3 bins
        if conf < 0.70:
            c_bin = 0
        elif conf < 0.90:
            c_bin = 1
        else:
            c_bin = 2

        exec_mask = (email_done << 2) | (url_done << 1) | sender_done
        budget_int = int(round(rem_budget * 3))

        return f"p{p_bin}_c{c_bin}_m{exec_mask}_b{budget_int}_x{contradiction}"

    def _get_q_values(self, state_key: str) -> list[float]:
        """Retrieve or initialize Q-values for a discrete state."""
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * NUM_ACTIONS
        return self.q_table[state_key]

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        explore: bool = False,
    ) -> int:
        """Select action with epsilon-greedy exploration or greedy exploitation."""
        if not valid_actions:
            return OrchestrationAction.STOP_AND_CLASSIFY

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(list(valid_actions)))

        state_key = self.discretize_state(state)
        q_vals = self._get_q_values(state_key)

        # Mask invalid actions with -inf
        best_val = -float("inf")
        best_action = valid_actions[0]
        for a in valid_actions:
            if q_vals[a] > best_val:
                best_val = q_vals[a]
                best_action = a

        return best_action

    def get_q_values_for_state(self, state: np.ndarray) -> list[float]:
        """Return Q-values for all actions in state."""
        state_key = self.discretize_state(state)
        return list(self._get_q_values(state_key))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_next_actions: Sequence[int],
    ) -> float:
        """Q-learning update step."""
        state_key = self.discretize_state(state)
        q_vals = self._get_q_values(state_key)
        old_val = q_vals[action]

        if done or not valid_next_actions:
            target = reward
        else:
            next_key = self.discretize_state(next_state)
            next_q = self._get_q_values(next_key)
            max_next_q = max(next_q[a] for a in valid_next_actions)
            target = reward + self.gamma * max_next_q

        td_error = target - old_val
        q_vals[action] = old_val + self.learning_rate * td_error
        return float(abs(td_error))

    def save(self, path: str | Path) -> None:
        """Save Q-table and metadata to JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_type": self.POLICY_TYPE,
            "version": self.VERSION,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "random_seed": self.random_seed,
            "q_table": self.q_table,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TabularQPolicy:
        """Load TabularQPolicy from JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Policy file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("policy_type") != cls.POLICY_TYPE:
            raise ValueError(
                f"Incompatible policy type: expected '{cls.POLICY_TYPE}', got '{payload.get('policy_type')}'"
            )

        policy = cls(
            learning_rate=payload["learning_rate"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            random_seed=payload.get("random_seed"),
        )
        policy.q_table = {k: list(v) for k, v in payload["q_table"].items()}
        return policy

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata summary."""
        return {
            "policy_type": self.POLICY_TYPE,
            "version": self.VERSION,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "table_entries": len(self.q_table),
        }


# ============================================================
# DEEP Q-NETWORK (DQN) IN PURE NUMPY
# ============================================================


class PureNumPyMLP:
    """
    2-Layer Multi-Layer Perceptron in pure NumPy.
    Layer 1: Linear + ReLU
    Layer 2: Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # He / Kaiming normal initialization
        w1_scale = np.sqrt(2.0 / input_dim)
        w2_scale = np.sqrt(2.0 / hidden_dim)

        self.w1 = (rng.standard_normal((hidden_dim, input_dim)) * w1_scale).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.standard_normal((output_dim, hidden_dim)) * w2_scale).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        Returns: (output Q-values, hidden activation)
        """
        # x shape: (input_dim,) or (batch_size, input_dim)
        z1 = np.dot(x, self.w1.T) + self.b1
        a1 = np.maximum(0.0, z1)  # ReLU
        q = np.dot(a1, self.w2.T) + self.b2
        return q, a1

    def copy_from(self, source: PureNumPyMLP) -> None:
        """Copy weights from another MLP."""
        self.w1 = source.w1.copy()
        self.b1 = source.b1.copy()
        self.w2 = source.w2.copy()
        self.b2 = source.b2.copy()


class DeepQPolicy(BaseRLPolicy):
    """
    Deep Q-Network (DQN) implemented in pure NumPy.
    Self-contained, fast, and dependency-free.

    Parameters
    ----------
    input_dim:
        State vector dimension (default 14).
    hidden_dim:
        Hidden layer units (default 32).
    learning_rate:
        Learning rate for gradient descent.
    gamma:
        Discount factor.
    epsilon:
        Exploration rate.
    target_update_freq:
        Frequency of updating the target network.
    random_seed:
        Seed for reproducibility.
    """

    POLICY_TYPE = "deep_q_network"
    VERSION = "1.0.0"

    def __init__(
        self,
        input_dim: int = STATE_DIMENSION,
        hidden_dim: int = 32,
        output_dim: int = NUM_ACTIONS,
        learning_rate: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        target_update_freq: int = 20,
        random_seed: int | None = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.target_update_freq = int(target_update_freq)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.step_counter = 0

        # Primary and target networks
        self.q_net = PureNumPyMLP(input_dim, hidden_dim, output_dim, self.rng)
        self.target_net = PureNumPyMLP(input_dim, hidden_dim, output_dim, self.rng)
        self.target_net.copy_from(self.q_net)

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        explore: bool = False,
    ) -> int:
        """Select action using epsilon-greedy or greedy choice."""
        if not valid_actions:
            return OrchestrationAction.STOP_AND_CLASSIFY

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(list(valid_actions)))

        q_vals, _ = self.q_net.forward(state)

        best_val = -float("inf")
        best_action = valid_actions[0]
        for a in valid_actions:
            if float(q_vals[a]) > best_val:
                best_val = float(q_vals[a])
                best_action = a

        return best_action

    def get_q_values_for_state(self, state: np.ndarray) -> list[float]:
        """Return Q-values for all actions."""
        q_vals, _ = self.q_net.forward(state)
        return [float(v) for v in q_vals]

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_next_actions: Sequence[int],
    ) -> float:
        """Single transition gradient step using backpropagation."""
        self.step_counter += 1

        # Target Q value
        if done or not valid_next_actions:
            target = float(reward)
        else:
            target_q_vals, _ = self.target_net.forward(next_state)
            max_next_q = max(float(target_q_vals[a]) for a in valid_next_actions)
            target = float(reward + self.gamma * max_next_q)

        # Forward pass on current state
        q_pred, a1 = self.q_net.forward(state)
        td_error = float(q_pred[action] - target)

        # Backpropagation
        # dL/dq = 2 * (q_pred[action] - target)
        dq = np.zeros(self.output_dim, dtype=np.float32)
        dq[action] = np.clip(td_error, -5.0, 5.0)  # Huber-like gradient clip

        # Gradients for layer 2
        dw2 = np.outer(dq, a1)
        db2 = dq

        # Gradients for layer 1
        da1 = np.dot(dq, self.q_net.w2)
        dz1 = da1 * (a1 > 0.0)  # ReLU derivative
        dw1 = np.outer(dz1, state)
        db1 = dz1

        # Update weights (SGD)
        lr = self.learning_rate
        self.q_net.w2 -= lr * dw2
        self.q_net.b2 -= lr * db2
        self.q_net.w1 -= lr * dw1
        self.q_net.b1 -= lr * db1

        # Periodically update target network
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.copy_from(self.q_net)

        return float(abs(td_error))

    def save(self, path: str | Path) -> None:
        """Save network weights and metadata."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_type": self.POLICY_TYPE,
            "version": self.VERSION,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "random_seed": self.random_seed,
            "weights": {
                "w1": self.q_net.w1.tolist(),
                "b1": self.q_net.b1.tolist(),
                "w2": self.q_net.w2.tolist(),
                "b2": self.q_net.b2.tolist(),
            },
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DeepQPolicy:
        """Load DeepQPolicy from file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("policy_type") != cls.POLICY_TYPE:
            raise ValueError(
                f"Incompatible policy type: expected '{cls.POLICY_TYPE}', got '{payload.get('policy_type')}'"
            )

        policy = cls(
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            output_dim=payload["output_dim"],
            learning_rate=payload["learning_rate"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            random_seed=payload.get("random_seed"),
        )
        weights = payload["weights"]
        policy.q_net.w1 = np.array(weights["w1"], dtype=np.float32)
        policy.q_net.b1 = np.array(weights["b1"], dtype=np.float32)
        policy.q_net.w2 = np.array(weights["w2"], dtype=np.float32)
        policy.q_net.b2 = np.array(weights["b2"], dtype=np.float32)
        policy.target_net.copy_from(policy.q_net)
        return policy

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata summary."""
        return {
            "policy_type": self.POLICY_TYPE,
            "version": self.VERSION,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "steps": self.step_counter,
        }


# ============================================================
# PRODUCTION ORCHESTRATION ADAPTER (RLEvidencePolicy)
# ============================================================


class RLEvidencePolicy:
    """
    Reinforcement Learning Evidence Policy for production orchestration.

    Wraps a trained BaseRLPolicy (DQN or Tabular Q) and provides:
    - Action selection with strict action masking.
    - Deterministic evaluation inference.
    - Seamless fallback to AdaptiveEvidencePolicy if model is absent.
    - Serialization of orchestration metadata.

    Parameters
    ----------
    policy:
        Underlying trained BaseRLPolicy (TabularQPolicy or DeepQPolicy).
    model_path:
        Optional path to load policy from.
    fallback_policy:
        Deterministic fallback policy if model is unavailable.
    confidence_threshold:
        Threshold used for fallback and state contradiction calculation.
    """

    POLICY_NAME = "rl_evidence_policy"

    def __init__(
        self,
        policy: BaseRLPolicy | None = None,
        model_path: str | Path | None = None,
        fallback_policy: AdaptiveEvidencePolicy | None = None,
        confidence_threshold: float = 0.90,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.confidence_threshold = float(confidence_threshold)
        self.fallback_policy = (
            fallback_policy
            if fallback_policy is not None
            else AdaptiveEvidencePolicy(
                confidence_threshold=confidence_threshold,
                min_evidence_count=0,
            )
        )

        self._policy: BaseRLPolicy | None = policy
        self._used_fallback = False

        if self._policy is None and self.model_path is not None:
            self._load_policy()

    def _load_policy(self) -> None:
        """Attempt to load policy from model_path."""
        if self.model_path is None or not self.model_path.exists():
            logger.warning(
                "RL model artifact not found at %s. Using deterministic fallback.",
                self.model_path,
            )
            self._used_fallback = True
            return

        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                header = json.load(f)
            ptype = header.get("policy_type")
            if ptype == DeepQPolicy.POLICY_TYPE:
                self._policy = DeepQPolicy.load(self.model_path)
            elif ptype == TabularQPolicy.POLICY_TYPE:
                self._policy = TabularQPolicy.load(self.model_path)
            else:
                raise ValueError(f"Unknown policy type in file: {ptype}")
            self._used_fallback = False
        except Exception as exc:
            logger.error("Failed to load RL model from %s: %s", self.model_path, exc)
            self._used_fallback = True

    @property
    def is_fallback_active(self) -> bool:
        """Return True if running in fallback mode."""
        return self._policy is None or self._used_fallback

    def select_action(
        self,
        results: Sequence[AgentResult],
        available_agents: Sequence[str],
        remaining_budget: int,
        max_agents: int = 3,
        executed_agents: Sequence[str] | None = None,
    ) -> tuple[OrchestrationAction, dict[str, Any]]:
        """
        Select the next orchestration action using the RL policy.

        Parameters
        ----------
        results:
            Agent results gathered so far in this analysis.
        available_agents:
            Names of agents eligible to run on this context.
        remaining_budget:
            Number of remaining agent executions allowed.
        max_agents:
            Total agent budget limit.
        executed_agents:
            Optional explicit list of executed or attempted agent names.

        Returns
        -------
        tuple[OrchestrationAction, dict[str, Any]]
            Selected action and diagnostic decision info.
        """
        if executed_agents is not None:
            executed_names = list(executed_agents)
        else:
            executed_names = [r.agent_name for r in results]
        valid_actions = get_valid_actions(executed_names, available_agents, remaining_budget)

        # Budget exhaustion check
        if remaining_budget <= 0:
            return OrchestrationAction.STOP_AND_CLASSIFY, {
                "reason": "Execution budget exhausted.",
                "policy": "budget_guard",
                "valid_actions": [OrchestrationAction.STOP_AND_CLASSIFY.value],
            }

        # If no agent actions valid, stop
        if not valid_actions:
            return OrchestrationAction.STOP_AND_CLASSIFY, {
                "reason": "No valid actions remain.",
                "policy": "safety_guard",
                "valid_actions": [],
            }

        # If running in fallback mode
        if self.is_fallback_active or self._policy is None:
            assessment = self.fallback_policy.assess(results)
            action = (
                OrchestrationAction.STOP_AND_CLASSIFY
                if assessment.sufficient
                else self._select_fallback_agent_action(executed_names, available_agents)
            )
            return action, {
                "reason": f"Fallback policy: {assessment.reason}",
                "policy": "fallback_adaptive",
                "used_fallback": True,
                "valid_actions": valid_actions,
            }

        # Encode state
        state_obj = encode_state(
            results=results,
            available_agents=available_agents,
            max_agents=max_agents,
            remaining_budget=remaining_budget,
            confidence_threshold=self.confidence_threshold,
        )

        action_idx = self._policy.select_action(
            state=state_obj.features,
            valid_actions=valid_actions,
            explore=False,  # deterministic during inference
        )
        action_enum = OrchestrationAction(action_idx)

        # Retrieve Q-values for decision trace
        q_values = []
        if hasattr(self._policy, "get_q_values_for_state"):
            q_values = self._policy.get_q_values_for_state(state_obj.features)

        info = {
            "reason": f"RL policy selected action {action_enum.name}.",
            "policy": self.POLICY_NAME,
            "policy_type": self._policy.get_metadata().get("policy_type", "unknown"),
            "action_name": action_enum.name,
            "valid_actions": valid_actions,
            "q_values": [round(float(q), 4) for q in q_values] if q_values else [],
            "used_fallback": False,
        }

        return action_enum, info

    def _select_fallback_agent_action(
        self,
        executed_names: Sequence[str],
        available_agents: Sequence[str],
    ) -> OrchestrationAction:
        """Select next agent in deterministic order (Email -> URL -> Sender)."""
        available_set = set(available_agents)
        executed_set = set(executed_names)

        order = [
            ("email_analysis_agent", OrchestrationAction.RUN_EMAIL_AGENT),
            ("url_analysis_agent", OrchestrationAction.RUN_URL_AGENT),
            ("sender_analysis_agent", OrchestrationAction.RUN_SENDER_AGENT),
        ]
        for agent_name, action in order:
            if agent_name in available_set and agent_name not in executed_set:
                return action
        return OrchestrationAction.STOP_AND_CLASSIFY

    def assess(self, results: Sequence[Any]) -> EvidencePolicyAssessment:
        """Compatibility method for orchestrator result assessment."""
        return self.fallback_policy.assess(results)

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for serialization."""
        base_meta = {
            "policy": self.POLICY_NAME,
            "confidence_threshold": self.confidence_threshold,
            "is_fallback_active": self.is_fallback_active,
            "model_path": str(self.model_path) if self.model_path else None,
        }
        if self._policy is not None:
            base_meta["rl_model_metadata"] = self._policy.get_metadata()
        return base_meta


__all__ = [
    "BaseRLPolicy",
    "TabularQPolicy",
    "PureNumPyMLP",
    "DeepQPolicy",
    "RLEvidencePolicy",
]
