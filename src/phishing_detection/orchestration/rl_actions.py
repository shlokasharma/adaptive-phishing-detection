"""
Reinforcement Learning Action Space for Phase 6.

Defines the discrete action space for adaptive multi-agent phishing orchestration.
Actions include running specialized detection agents or terminating analysis
to classify with the evidence collected so far.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Sequence


class OrchestrationAction(IntEnum):
    """Discrete actions available to the RL orchestration policy."""

    RUN_EMAIL_AGENT = 0
    RUN_URL_AGENT = 1
    RUN_SENDER_AGENT = 2
    STOP_AND_CLASSIFY = 3

    @classmethod
    def from_string(cls, name: str) -> OrchestrationAction:
        """Parse an action from string name."""
        name_clean = name.strip().upper()
        mapping = {
            "RUN_EMAIL_AGENT": cls.RUN_EMAIL_AGENT,
            "EMAIL": cls.RUN_EMAIL_AGENT,
            "RUN_URL_AGENT": cls.RUN_URL_AGENT,
            "URL": cls.RUN_URL_AGENT,
            "RUN_SENDER_AGENT": cls.RUN_SENDER_AGENT,
            "SENDER": cls.RUN_SENDER_AGENT,
            "STOP_AND_CLASSIFY": cls.STOP_AND_CLASSIFY,
            "STOP": cls.STOP_AND_CLASSIFY,
        }
        if name_clean not in mapping:
            raise ValueError(
                f"Unknown action name '{name}'. Expected one of {list(mapping.keys())}"
            )
        return mapping[name_clean]

    @property
    def action_name(self) -> str:
        """Return standardized human-readable name."""
        return self.name

    @property
    def target_agent_name(self) -> str | None:
        """Return the agent name targeted by this action, or None if terminal."""
        if self == OrchestrationAction.RUN_EMAIL_AGENT:
            return "email_analysis_agent"
        if self == OrchestrationAction.RUN_URL_AGENT:
            return "url_analysis_agent"
        if self == OrchestrationAction.RUN_SENDER_AGENT:
            return "sender_analysis_agent"
        return None

    @property
    def is_terminal(self) -> bool:
        """Return True if this action stops analysis and triggers classification."""
        return self == OrchestrationAction.STOP_AND_CLASSIFY


# Number of discrete actions
NUM_ACTIONS = len(OrchestrationAction)
ACTION_NAMES = [a.name for a in OrchestrationAction]


def get_agent_action_map() -> dict[str, OrchestrationAction]:
    """Map agent names to their corresponding actions."""
    return {
        "email_analysis_agent": OrchestrationAction.RUN_EMAIL_AGENT,
        "url_analysis_agent": OrchestrationAction.RUN_URL_AGENT,
        "sender_analysis_agent": OrchestrationAction.RUN_SENDER_AGENT,
    }


def compute_action_mask(
    executed_agents: Sequence[str],
    available_agents: Sequence[str],
    remaining_budget: int,
) -> list[bool]:
    """
    Compute a boolean mask indicating which actions are valid.

    Parameters
    ----------
    executed_agents:
        Names of agents that have already been executed in this episode.
    available_agents:
        Names of agents eligible to run on the current input context.
    remaining_budget:
        Number of remaining agent executions allowed.

    Returns
    -------
    list[bool]
        Boolean list of length NUM_ACTIONS where True means the action is valid.
    """
    mask = [False] * NUM_ACTIONS
    executed_set = set(executed_agents)
    available_set = set(available_agents)

    # Agent actions: valid only if available, not yet executed, and budget > 0
    if remaining_budget > 0:
        if "email_analysis_agent" in available_set and "email_analysis_agent" not in executed_set:
            mask[OrchestrationAction.RUN_EMAIL_AGENT] = True

        if "url_analysis_agent" in available_set and "url_analysis_agent" not in executed_set:
            mask[OrchestrationAction.RUN_URL_AGENT] = True

        if "sender_analysis_agent" in available_set and "sender_analysis_agent" not in executed_set:
            mask[OrchestrationAction.RUN_SENDER_AGENT] = True

    # STOP_AND_CLASSIFY is always valid (the agent can always decide to stop)
    mask[OrchestrationAction.STOP_AND_CLASSIFY] = True

    return mask


def get_valid_actions(
    executed_agents: Sequence[str],
    available_agents: Sequence[str],
    remaining_budget: int,
) -> list[int]:
    """Return indices of valid actions."""
    mask = compute_action_mask(executed_agents, available_agents, remaining_budget)
    return [i for i, valid in enumerate(mask) if valid]


__all__ = [
    "OrchestrationAction",
    "NUM_ACTIONS",
    "ACTION_NAMES",
    "get_agent_action_map",
    "compute_action_mask",
    "get_valid_actions",
]
