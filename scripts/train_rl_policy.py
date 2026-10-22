"""
Train Reinforcement Learning Policy for Adaptive Evidence Orchestration (Phase 6).

Trains a discrete action orchestration policy (DQN or Tabular Q) to balance
detection accuracy, false-negative penalties, and computational execution costs.
Uses real agent models and split datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.orchestration.rl_actions import (
    ACTION_NAMES,
    OrchestrationAction,
)
from phishing_detection.orchestration.rl_environment import (
    PhishingRLEnvironment,
)
from phishing_detection.orchestration.rl_policy import (
    DeepQPolicy,
    TabularQPolicy,
)
from phishing_detection.orchestration.rl_reward import (
    RewardConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

URL_REGEX = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")
EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def create_agent_context_from_row(row: pd.Series) -> tuple[AgentContext, int]:
    """Construct an AgentContext and ground-truth label from a dataset row."""
    text = str(row.get("clean_text", ""))
    label = int(row.get("label", 0))

    # Extract URL if present
    found_urls = URL_REGEX.findall(text)
    url = found_urls[0] if found_urls else ""

    # Extract sender if present
    found_emails = EMAIL_REGEX.findall(text)
    sender = found_emails[0] if found_emails else ""

    metadata = {}
    if url:
        metadata["url_features"] = {"url_length": len(url)}

    context = AgentContext(
        input_text=text,
        url=url,
        sender=sender,
        headers={"spf": "pass"} if sender else {},
        metadata=metadata,
    )
    return context, label


def build_agents(project_root: Path) -> tuple[EmailAnalysisAgent, URLAnalysisAgent, SenderAnalysisAgent]:
    """Instantiate agents with trained baseline models."""
    email_model_path = project_root / "experiments" / "baselines" / "models" / "email_linear_svm.pkl"
    email_tfidf_path = project_root / "experiments" / "baselines" / "models" / "email_linear_svm_tfidf_extractor.pkl"

    url_model_path = project_root / "experiments" / "baselines" / "models" / "url_gradient_boosting.pkl"
    url_prep_path = project_root / "experiments" / "baselines" / "models" / "url_gradient_boosting_feature_extractor.pkl"

    email_agent = EmailAnalysisAgent(
        model_path=email_model_path,
        extractor_path=email_tfidf_path,
    )

    url_agent = URLAnalysisAgent(
        model_path=url_model_path,
        preprocessor_path=url_prep_path,
    )

    sender_agent = SenderAnalysisAgent()

    return email_agent, url_agent, sender_agent


def train_policy(
    algorithm: str = "dqn",
    episodes: int = 150,
    learning_rate: float = 0.01,
    epsilon_start: float = 0.3,
    epsilon_min: float = 0.05,
    gamma: float = 0.95,
    max_agents: int = 3,
    seed: int = 42,
    output_path: Path | None = None,
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Run RL training loop and save trained policy."""
    rng = np.random.default_rng(seed)

    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "splits" / "email" / "train.csv"

    if output_path is None:
        out_dir = PROJECT_ROOT / "experiments" / "models" / "rl"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{algorithm}_policy.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data from %s...", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    # Load sample subset for training
    df = pd.read_csv(data_path)
    valid_rows = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]
    total_samples = len(valid_rows)
    logger.info("Loaded %d valid samples.", total_samples)

    # Initialize agents
    email_agent, url_agent, sender_agent = build_agents(PROJECT_ROOT)

    # Reward configuration with security-conscious false negative penalty
    reward_config = RewardConfig(
        reward_tp=10.0,
        reward_tn=10.0,
        penalty_fp=10.0,
        penalty_fn=25.0,  # Missed phishing is heavily penalized
        step_cost=0.5,    # Encourage concise analysis
        penalty_premature_stop=5.0,
        evidence_strength_weight=1.0,
    )

    env = PhishingRLEnvironment(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        max_agents=max_agents,
        reward_config=reward_config,
    )

    # Initialize policy
    if algorithm == "dqn":
        policy = DeepQPolicy(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon_start,
            random_seed=seed,
        )
    elif algorithm == "tabular":
        policy = TabularQPolicy(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon_start,
            random_seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'dqn' or 'tabular'.")

    logger.info("Starting %s training for %d episodes (seed=%d)...", algorithm.upper(), episodes, seed)
    start_time = time.perf_counter()

    episode_rewards = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    outcomes = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "premature": 0}

    # Sample rows for episodes
    sampled_indices = rng.choice(total_samples, size=episodes, replace=True)

    for ep in range(episodes):
        # Epsilon decay
        decay = max(epsilon_min, epsilon_start * (1.0 - ep / episodes))
        policy.epsilon = decay

        row = valid_rows.iloc[int(sampled_indices[ep])]
        context, true_label = create_agent_context_from_row(row)

        state, info = env.reset(context, ground_truth=true_label)
        total_ep_reward = 0.0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action_idx = policy.select_action(state, valid_actions, explore=True)
            action_enum = OrchestrationAction(action_idx)
            action_counts[action_enum.name] += 1

            step_res = env.step(action_idx)
            next_state = step_res.state
            reward = step_res.reward
            done = step_res.done

            valid_next = env.get_valid_actions() if not done else []

            # Update policy
            policy.update(
                state=state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_next_actions=valid_next,
            )

            total_ep_reward += reward
            state = next_state

        episode_rewards.append(total_ep_reward)

        # Track outcome
        breakdown = step_res.info.get("terminal_breakdown", {})
        oc = breakdown.get("outcome", "unknown")
        if oc in outcomes:
            outcomes[oc] += 1
        if len(env.results) == 0:
            outcomes["premature"] += 1

        if (ep + 1) % max(1, (episodes // 5)) == 0 or ep == episodes - 1:
            avg_rew = np.mean(episode_rewards[-20:])
            logger.info(
                "Episode %3d/%d | Epsilon: %.3f | Recent Mean Reward: %6.2f | TP:%d TN:%d FP:%d FN:%d",
                ep + 1,
                episodes,
                decay,
                avg_rew,
                outcomes["TP"],
                outcomes["TN"],
                outcomes["FP"],
                outcomes["FN"],
            )

    train_time = time.perf_counter() - start_time
    logger.info("Training completed in %.2f seconds.", train_time)

    # Save policy
    policy.save(output_path)
    logger.info("Saved trained policy artifact to: %s", output_path)

    summary = {
        "algorithm": algorithm,
        "episodes": episodes,
        "training_time_seconds": round(train_time, 2),
        "mean_episode_reward": round(float(np.mean(episode_rewards)), 2),
        "final_20_reward": round(float(np.mean(episode_rewards[-20:])), 2),
        "action_counts": action_counts,
        "outcomes": outcomes,
        "output_path": str(output_path),
        "metadata": policy.get_metadata(),
    }

    # Save summary metadata
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved metadata summary to: %s", meta_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Phase 6 RL policy for phishing orchestration.")
    parser.add_argument("--algorithm", choices=["dqn", "tabular"], default="dqn", help="RL algorithm to train.")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Initial exploration rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-agents", type=int, default=3, help="Max agent budget.")
    parser.add_argument("--output", type=str, default=None, help="Output path for trained policy JSON.")
    parser.add_argument("--data", type=str, default=None, help="Path to training CSV.")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    data_path = Path(args.data) if args.data else None

    summary = train_policy(
        algorithm=args.algorithm,
        episodes=args.episodes,
        learning_rate=args.lr,
        epsilon_start=args.epsilon,
        max_agents=args.max_agents,
        seed=args.seed,
        output_path=out_path,
        data_path=data_path,
    )

    print("\n" + "=" * 60)
    print("PHASE 6 REINFORCEMENT LEARNING TRAINING COMPLETE")
    print("=" * 60)
    print(f"Algorithm:       {summary['algorithm'].upper()}")
    print(f"Episodes:        {summary['episodes']}")
    print(f"Training Time:   {summary['training_time_seconds']}s")
    print(f"Mean Reward:     {summary['mean_episode_reward']}")
    print(f"Saved Artifact:  {summary['output_path']}")
    print(f"Action Counts:   {summary['action_counts']}")
    print(f"Outcomes:        {summary['outcomes']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
