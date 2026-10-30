#!/usr/bin/env python3
"""
Training script for Drakarys RL agents.

Supports:
- Q-learning for discrete actions
- Deep Q-learning (DQN) via --agent dqn (requires torch)
- Policy gradient methods
- Random baseline comparison
- Model saving/loading
"""

import argparse
import os
import sys
import json
import pickle
import random
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from eth_account import Account

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from drakarys_env import DrakarysEnv, N_PLAYERS, N_TEAMS, N_ROUNDS, DELTA_X_RANGE, DELTA_Y_RANGE

# Constants
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01


class QLearningAgent:
    """Simple Q-learning agent for Drakarys"""
    
    def __init__(self, state_size: int, action_size: int, private_key: str = None):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration_rate = EXPLORATION_RATE
        
        # Generate or use provided private key for signing
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = Account.create()
        self.private_key = self.account.key.hex()
        self.address = self.account.address
        
    def get_state_key(self, state: Dict) -> str:
        """Convert state to a hashable key"""
        # Simplified state representation
        ball_x = state["ball_holder"][1]
        ball_y = 0
        score_diff = state["score"][0] - state["score"][1]
        state_idx = state["state_idx"][0]
        
        return f"{ball_x}_{ball_y}_{score_diff}_{state_idx}"
    
    def get_action(self, state: Dict) -> Dict:
        """Choose an action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if random.random() < self.exploration_rate:
            # Random action
            return self._random_action()
        else:
            # Greedy action
            action_idx = np.argmax(self.q_table[state_key])
            return self._action_from_index(action_idx)
    
    def _random_action(self) -> Dict:
        """Generate a random action"""
        return {
            "player_deltas": [
                [random.randint(-DELTA_X_RANGE, DELTA_X_RANGE),
                 random.randint(-DELTA_Y_RANGE, DELTA_Y_RANGE)]
                for _ in range(N_PLAYERS)
            ],
            "pass_target": random.randint(0, N_PLAYERS-1),
            "shoot": random.randint(0, 1),
        }
    
    def _action_from_index(self, idx: int) -> Dict:
        """Convert action index to action dict"""
        # Simplified: map index to pass/shoot
        shoot = idx % 2
        pass_target = (idx // 2) % N_PLAYERS
        deltas = [[0, 0]] * N_PLAYERS
        
        return {
            "player_deltas": deltas,
            "pass_target": pass_target,
            "shoot": shoot,
        }
    
    def learn(self, state: Dict, action: Dict, reward: float, next_state: Dict) -> None:
        """Update Q-table"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_idx = self._action_to_index(action)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        old_value = self.q_table[state_key][action_idx]
        next_max = np.max(self.q_table[next_state_key])
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        self.q_table[state_key][action_idx] = new_value
        
        # Decay exploration rate
        self.exploration_rate = max(
            MIN_EXPLORATION_RATE,
            self.exploration_rate * EXPLORATION_DECAY
        )
    
    def _action_to_index(self, action: Dict) -> int:
        """Convert action dict to index"""
        shoot = action["shoot"]
        pass_target = action["pass_target"]
        return shoot + pass_target * 2
    
    def save(self, filepath: str) -> None:
        """Save the Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath: str) -> None:
        """Load the Q-table"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)


class RandomAgent:
    """Baseline random agent"""
    
    def __init__(self, private_key: str = None):
        # Generate or use provided private key for signing
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = Account.create()
        self.private_key = self.account.key.hex()
        self.address = self.account.address
    
    def get_action(self, state: Dict) -> Dict:
        return {
            "player_deltas": [
                [random.randint(-DELTA_X_RANGE, DELTA_X_RANGE),
                 random.randint(-DELTA_Y_RANGE, DELTA_Y_RANGE)]
                for _ in range(N_PLAYERS)
            ],
            "pass_target": random.randint(0, N_PLAYERS-1),
            "shoot": random.randint(0, 1),
        }


def train_agent(
    env: DrakarysEnv,
    agent,
    episodes: int = 1000,
    save_interval: int = 100,
    model_path: str = "q_table.pkl",
):
    """Train an agent on the Drakarys environment"""
    episode_rewards = []
    win_rates = []
    wins = 0
    
    # Set the environment to use this agent's private key
    env.player_account = agent.account
    env.player_key = agent.private_key
    env.player_address = agent.address
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            if hasattr(agent, 'learn'):
                agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        # Track wins (assuming positive reward = win)
        if total_reward > 0:
            wins += 1
        
        win_rate = wins / (episode + 1)
        win_rates.append(win_rate)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Average reward: {np.mean(episode_rewards[-100:]):.3f}")
            print(f"  Win rate: {win_rate:.3f}")
            if hasattr(agent, 'exploration_rate'):
                print(f"  Exploration rate: {agent.exploration_rate:.3f}")
            print()
        
        if (episode + 1) % save_interval == 0 and hasattr(agent, 'save'):
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    return episode_rewards, win_rates


def evaluate_agent(env: DrakarysEnv, agent, episodes: int = 100):
    """Evaluate agent performance (exploration disabled)"""
    wins = 0
    total_rewards = []
    
    # Set the environment to use this agent's private key
    env.player_account = agent.account
    env.player_key = agent.private_key
    env.player_address = agent.address
    
    # Disable exploration for a fair greedy-policy evaluation
    saved_exploration = getattr(agent, "exploration_rate", None)
    if saved_exploration is not None:
        agent.exploration_rate = 0.0
    
    try:
        for _ in range(episodes):
            state, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = agent.get_action(state)
                state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if episode_reward > 0:
                wins += 1
    finally:
        if saved_exploration is not None:
            agent.exploration_rate = saved_exploration
    
    win_rate = wins / episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"Evaluation over {episodes} episodes:")
    print(f"  Win rate: {win_rate:.3f}")
    print(f"  Average reward: {avg_reward:.3f}")
    
    return win_rate, avg_reward


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description="Train Drakarys RL agents")
    parser.add_argument(
        "--agent",
        choices=["q", "dqn"],
        default="q",
        help="Agent type to train: tabular q-learning or deep q-network",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    
    env = DrakarysEnv()
    
    if args.agent == "dqn":
        from dqn_agent import DQNAgent
        agent = DQNAgent(private_key=os.environ.get("DQN_AGENT_KEY"))
        model_path = args.model_path or "dqn_model.pt"
    else:
        agent = QLearningAgent(
            state_size=1000,
            action_size=N_PLAYERS * 2,
            private_key=os.environ.get("Q_AGENT_KEY"),
        )
        model_path = args.model_path or "q_table.pkl"
    
    random_agent = RandomAgent(private_key=os.environ.get("RANDOM_AGENT_KEY"))
    
    print(f"Agent address: {agent.address}")
    print(f"Random Agent address: {random_agent.address}")
    
    print(f"Training {args.agent} agent...")
    rewards, win_rates = train_agent(
        env, agent, episodes=args.episodes, model_path=model_path
    )
    
    print(f"\nEvaluating {args.agent} agent...")
    evaluate_agent(env, agent, episodes=args.eval_episodes)
    
    print("\nEvaluating random agent...")
    evaluate_agent(env, random_agent, episodes=args.eval_episodes)
    
    env.close()


if __name__ == "__main__":
    main()
