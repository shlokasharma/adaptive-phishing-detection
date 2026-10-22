"""
Reinforcement Learning Reward Function for Phase 6.

Computes the reward signal during training to guide the policy toward:
- High classification accuracy (prioritizing false negative avoidance).
- Minimal execution cost and latency.
- Sufficient evidence gathering before stopping.

Ground-truth labels are strictly used during training reward calculation
and are never passed into the inference state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from phishing_detection.agents.confidence import (
    calculate_aggregate_evidence_strength,
    calculate_confidence,
)
from phishing_detection.agents.result import AgentResult
from phishing_detection.orchestration.rl_actions import OrchestrationAction


@dataclass
class RewardConfig:
    """
    Configuration parameters for the RL orchestration reward function.

    Parameters
    ----------
    reward_tp:
        Reward for correctly identifying phishing (True Positive).
    reward_tn:
        Reward for correctly identifying legitimate (True Negative).
    penalty_fp:
        Penalty (positive number, subtracted) for classifying legitimate as phishing.
    penalty_fn:
        Penalty (positive number, subtracted) for missing phishing (False Negative).
        Typically set higher than FP due to severe security threat of missed attacks.
    step_cost:
        Cost incurred per agent execution (latency and computation penalty).
    penalty_premature_stop:
        Penalty for stopping with zero agents executed or near-zero evidence.
    evidence_strength_weight:
        Weight for bonus awarded based on aggregate evidence strength at classification.
    invalid_action_penalty:
        Penalty applied when the policy chooses an ineligible or masked action.
    """

    reward_tp: float = 10.0
    reward_tn: float = 10.0
    penalty_fp: float = 10.0
    penalty_fn: float = 25.0
    step_cost: float = 0.5
    penalty_premature_stop: float = 5.0
    evidence_strength_weight: float = 1.0
    invalid_action_penalty: float = 5.0

    def to_dict(self) -> dict[str, float]:
        """Convert configuration to dictionary."""
        return {
            "reward_tp": self.reward_tp,
            "reward_tn": self.reward_tn,
            "penalty_fp": self.penalty_fp,
            "penalty_fn": self.penalty_fn,
            "step_cost": self.step_cost,
            "penalty_premature_stop": self.penalty_premature_stop,
            "evidence_strength_weight": self.evidence_strength_weight,
            "invalid_action_penalty": self.invalid_action_penalty,
        }


def calculate_step_reward(
    action: OrchestrationAction | int,
    is_valid: bool,
    config: RewardConfig,
) -> float:
    """
    Calculate intermediate reward for an agent execution step.

    Parameters
    ----------
    action:
        The action taken by the policy.
    is_valid:
        Whether the action was permissible in the current state.
    config:
        Reward configuration.

    Returns
    -------
    float
        Immediate step reward.
    """
    if not is_valid:
        return -config.invalid_action_penalty

    action_enum = OrchestrationAction(action)
    if action_enum.is_terminal:
        # Terminal action reward will be calculated by calculate_terminal_reward
        return 0.0

    # Agent execution incurs step cost
    return -config.step_cost


def calculate_terminal_reward(
    results: Sequence[AgentResult],
    ground_truth: int,
    config: RewardConfig,
) -> tuple[float, dict[str, Any]]:
    """
    Calculate the terminal reward when analysis stops and final decision is made.

    Parameters
    ----------
    results:
        Agent results collected in the episode.
    ground_truth:
        True binary label (1 = phishing, 0 = legitimate).
    config:
        Reward configuration.

    Returns
    -------
    tuple[float, dict[str, Any]]
        Total terminal reward and breakdown dictionary.
    """
    if ground_truth not in (0, 1):
        raise ValueError(f"ground_truth must be 0 or 1, got {ground_truth}")

    # Determine final prediction
    if results:
        probs = [float(r.phishing_probability) for r in results]
        combined_prob = sum(probs) / len(probs)
        pred_label = int(combined_prob >= 0.5)
        conf = calculate_confidence(combined_prob)
        all_evidence = [ev for r in results for ev in r.evidence]
        ev_strength = calculate_aggregate_evidence_strength(all_evidence)
    else:
        # Stopped without running any agent
        combined_prob = 0.5
        pred_label = 0
        conf = 0.5
        ev_strength = 0.0

    classification_reward = 0.0
    outcome = "unknown"

    if pred_label == 1 and ground_truth == 1:
        # True Positive
        classification_reward = config.reward_tp * conf
        outcome = "TP"
    elif pred_label == 0 and ground_truth == 0:
        # True Negative
        classification_reward = config.reward_tn * conf
        outcome = "TN"
    elif pred_label == 1 and ground_truth == 0:
        # False Positive
        classification_reward = -config.penalty_fp
        outcome = "FP"
    elif pred_label == 0 and ground_truth == 1:
        # False Negative (missed phishing - high security cost)
        classification_reward = -config.penalty_fn
        outcome = "FN"

    # Premature stop penalty: if no agents were executed
    premature_penalty = 0.0
    if len(results) == 0:
        premature_penalty = -config.penalty_premature_stop

    # Evidence strength bonus: reward collecting reliable evidence
    evidence_bonus = config.evidence_strength_weight * ev_strength

    total_reward = classification_reward + premature_penalty + evidence_bonus

    breakdown = {
        "outcome": outcome,
        "predicted_label": pred_label,
        "ground_truth": ground_truth,
        "combined_probability": round(combined_prob, 4),
        "confidence": round(conf, 4),
        "classification_reward": round(classification_reward, 4),
        "premature_penalty": round(premature_penalty, 4),
        "evidence_bonus": round(evidence_bonus, 4),
        "total_terminal_reward": round(total_reward, 4),
    }

    return total_reward, breakdown


__all__ = [
    "RewardConfig",
    "calculate_step_reward",
    "calculate_terminal_reward",
]
