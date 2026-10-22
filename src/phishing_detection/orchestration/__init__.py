"""
Adaptive evidence orchestration with Reinforcement Learning support.

Phase 3: Deterministic orchestration across specialized phishing detection agents.
Phase 5: Adaptive evidence policy evaluating combined multi-agent evidence.
Phase 6: Reinforcement Learning (RL) policy for dynamic analysis action selection.
"""

from phishing_detection.orchestration.adaptive_policy import (
    AdaptiveEvidencePolicy,
    EvidencePolicyAssessment,
)
from phishing_detection.orchestration.orchestrator import (
    OrchestrationResult,
    OrchestrationStep,
    OrchestratorAgent,
)
from phishing_detection.explainability.comparison import (
    XAIComparisonResult,
    compare_explanations,
)
from phishing_detection.orchestration.rl_actions import (
    NUM_ACTIONS,
    ACTION_NAMES,
    OrchestrationAction,
    compute_action_mask,
    get_valid_actions,
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

__all__ = [
    # Phase 3 & Phase 5
    "AdaptiveEvidencePolicy",
    "EvidencePolicyAssessment",
    "OrchestrationResult",
    "OrchestrationStep",
    "OrchestratorAgent",
    # Phase 4
    "XAIComparisonResult",
    "compare_explanations",
    # Phase 6 RL
    "OrchestrationAction",
    "NUM_ACTIONS",
    "ACTION_NAMES",
    "compute_action_mask",
    "get_valid_actions",
    "RLState",
    "encode_state",
    "STATE_DIMENSION",
    "STATE_FEATURE_NAMES",
    "RewardConfig",
    "calculate_step_reward",
    "calculate_terminal_reward",
    "BaseRLPolicy",
    "TabularQPolicy",
    "PureNumPyMLP",
    "DeepQPolicy",
    "RLEvidencePolicy",
    "PhishingRLEnvironment",
    "StepResult",
]
