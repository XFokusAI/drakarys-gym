#!/usr/bin/env python3
"""
Training script for the Drakarys DQN agent.

Each episode is a full 20-round match on the local anvil chain (2
transactions per round: commit + reveal), so episodes are slow - size
--episodes accordingly. Learning only starts once the replay buffer
holds MIN_REPLAY_SIZE transitions (~13 episodes of warmup).

Checkpoints store policy/target/optimizer state plus epsilon, so
training can be resumed with --resume.

Usage:
    python3 train_dqn.py --episodes 500
    python3 train_dqn.py --resume dqn_model.pt --episodes 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from drakarys_env import DrakarysEnv, N_ROUNDS
from dqn_agent import DQNAgent
from train_agent import RandomAgent, evaluate_agent


def train(
    env: DrakarysEnv,
    agent: DQNAgent,
    episodes: int,
    model_path: str,
    save_every: int = 50,
    eval_every: int = 0,
    eval_episodes: int = 20,
    metrics_path: str = None,
):
    """Train the DQN agent, checkpointing and evaluating periodically."""
    env.player_account = agent.account
    env.player_key = agent.private_key
    env.player_address = agent.address

    episode_rewards = []
    metrics = {
        "episodes": [], "avg_reward": [], "win_rate": [],
        "epsilon": [], "loss": [], "score_for": [], "score_against": [],
    }
    wins = 0
    start = time.time()

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = truncated = False

        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

        # final on-chain score, from the agent's perspective
        score = state["score"]
        s_for, s_against = score[env.player_team], score[1 - env.player_team]

        recent = episode_rewards[-100:]
        win_rate = wins / (episode + 1)
        loss = agent.last_loss

        metrics["episodes"].append(episode + 1)
        metrics["avg_reward"].append(float(np.mean(recent)))
        metrics["win_rate"].append(win_rate)
        metrics["epsilon"].append(agent.exploration_rate)
        metrics["loss"].append(loss)
        metrics["score_for"].append(s_for)
        metrics["score_against"].append(s_against)

        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start
            loss_str = f"{loss:.4f}" if loss is not None else "warmup"
            print(
                f"episode {episode + 1:>5}/{episodes}"
                f" | avg_reward(100) {np.mean(recent):+.3f}"
                f" | win_rate {win_rate:.3f}"
                f" | eps {agent.exploration_rate:.3f}"
                f" | loss {loss_str}"
                f" | goals {sum(metrics['score_for'][-10:])}-{sum(metrics['score_against'][-10:])}(10)"
                f" | replay {len(agent.replay)}"
                f" | {elapsed / 60:.1f}min"
            )

        if save_every and (episode + 1) % save_every == 0:
            agent.save(model_path)
            print(f"  checkpoint saved -> {model_path}")

        if eval_every and (episode + 1) % eval_every == 0:
            print(f"  greedy eval ({eval_episodes} episodes):")
            evaluate_agent(env, agent, episodes=eval_episodes)

    agent.save(model_path)
    print(f"final checkpoint saved -> {model_path}")

    if metrics_path:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"metrics saved -> {metrics_path}")

    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Train the Drakarys DQN agent")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model-path", default="dqn_model.pt")
    parser.add_argument("--metrics-path", default="dqn_metrics.json")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="run a greedy eval every N episodes (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=50)
    parser.add_argument("--resume", default=None,
                        help="checkpoint path to resume training from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda")
    args = parser.parse_args()

    env = DrakarysEnv(contract_path='./artifacts/Drakarys_GAME_0.json')

    agent = DQNAgent(
        private_key=os.environ.get("DQN_AGENT_KEY"),
        player_team=env.player_team,
        device=args.device,
        seed=args.seed,
    )
    if args.resume:
        agent.load(args.resume)
        print(f"resumed from {args.resume} "
              f"(eps={agent.exploration_rate:.3f}, steps={agent.learn_steps})")

    print(f"DQN agent address: {agent.address}")
    print(f"device: {agent.device} | actions: {agent.action_size} | state_dim: {agent.state_dim}")

    train(
        env, agent,
        episodes=args.episodes,
        model_path=args.model_path,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        metrics_path=args.metrics_path,
    )

    print("\nFinal greedy evaluation:")
    evaluate_agent(env, agent, episodes=args.final_eval_episodes)

    print("\nRandom-agent baseline:")
    random_agent = RandomAgent(private_key=os.environ.get("RANDOM_AGENT_KEY"))
    evaluate_agent(env, random_agent, episodes=args.final_eval_episodes)

    env.close()


if __name__ == "__main__":
    main()
