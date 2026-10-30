#!/usr/bin/env python3
"""
Deep Q-Network (Double DQN) agent for Drakarys.

The contract action space (11 players x (dx, dy) deltas + pass/shoot command)
is far too large to explore directly, so the agent acts over a small set of
discrete macro-actions: a team-level movement pattern combined with a ball
command (none / pass / shoot).

Requires PyTorch: pip install torch
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None

from eth_account import Account

from drakarys_env import (
    FIELD_W, FIELD_H, N_PLAYERS, N_TEAMS, N_ROUNDS,
    DELTA_X_RANGE, DELTA_Y_RANGE,
)

# Hyperparameters
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
MIN_REPLAY_SIZE = 256
TARGET_UPDATE_INTERVAL = 250    # gradient steps between target-net syncs
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.9995      # per learn() call (i.e. per env step)
MIN_EXPLORATION_RATE = 0.05

# Team-level movement patterns -> per-player delta vectors
TEAM_MOVES = ["hold", "attack", "defend", "push_up", "push_down", "chase_ball"]

# Ball commands -> (pass, shoot) flags on the move
COMMANDS = ["none", "pass", "shoot"]

N_MOVES = len(TEAM_MOVES)
N_COMMANDS = len(COMMANDS)
N_ACTIONS = N_MOVES * N_COMMANDS

# Encoded observation dimension:
#   my_pos (22) + opp_pos (22) + ball (2) + score (2) + state_idx (1)
#   + my_stats (22) + opp_stats (22) = 93
STATE_DIM = 93


def encode_state(state: Dict, player_team: int = 0) -> np.ndarray:
    """Flatten an observation dict into a normalized vector from the
    agent's perspective (agent's team always encoded first)."""
    t, o = player_team, 1 - player_team

    pos = np.asarray(state["player_positions"], dtype=np.float32)
    field = np.array([FIELD_W, FIELD_H], dtype=np.float32)
    stats = np.asarray(
        state.get("player_stats", np.zeros((N_TEAMS, N_PLAYERS, 2))),
        dtype=np.float32,
    ) / 99.0

    bh_team, bh_player = int(state["ball_holder"][0]), int(state["ball_holder"][1])
    ball = np.array(
        [1.0 if bh_team == t else 0.0, bh_player],
        dtype=np.float32,
    )
    score = np.asarray(state["score"], dtype=np.float32)
    rel_score = np.array([score[t], score[o]], dtype=np.float32) / N_ROUNDS
    idx = np.array([state["state_idx"][0] / N_ROUNDS], dtype=np.float32)

    return np.concatenate([
        (pos[t] / field).flatten(),
        (pos[o] / field).flatten(),
        ball,
        rel_score,
        idx,
        stats[t].flatten(),
        stats[o].flatten(),
    ])


class DQNetwork(nn.Module):
    """MLP approximating Q(s, a) over macro-actions."""

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done) -> None:
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            np.asarray(s),
            np.asarray(a),
            np.asarray(r, dtype=np.float32),
            np.asarray(s_next),
            np.asarray(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Double DQN agent with experience replay and a target network.

    Same interface as QLearningAgent in train_agent.py:
    get_action(state) -> action dict, learn(state, action, reward, next_state),
    save/load(filepath), account/private_key/address for move signing.
    """

    def __init__(
        self,
        state_size: int = None,      # accepted for signature parity; derived internally
        action_size: int = None,
        private_key: str = None,
        player_team: int = 0,
        device: str = None,
        seed: int = None,
    ):
        if torch is None:
            raise ImportError("DQNAgent requires PyTorch: pip install torch")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.player_team = player_team
        self.state_dim = STATE_DIM
        self.action_size = N_ACTIONS
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = DQNetwork(self.state_dim, self.action_size).to(self.device)
        self.target_net = DQNetwork(self.state_dim, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay = ReplayBuffer(REPLAY_CAPACITY)

        self.exploration_rate = EXPLORATION_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.learn_steps = 0
        self.last_loss = None

        # Account for signing commitments (same convention as other agents)
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = Account.create()
        self.private_key = self.account.key.hex()
        self.address = self.account.address

        # Maps the action dict we returned to its macro-action index
        self._last_action: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, state: Dict) -> Dict:
        """Epsilon-greedy action selection."""
        if random.random() < self.exploration_rate:
            idx = random.randrange(self.action_size)
        else:
            idx = self._greedy_index(state)

        action = self._action_from_index(idx, state)
        self._last_action = (id(action), idx)
        return action

    def _greedy_index(self, state: Dict) -> int:
        with torch.no_grad():
            s = torch.as_tensor(
                encode_state(state, self.player_team),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            return int(self.policy_net(s).argmax(1).item())

    def _action_from_index(self, idx: int, state: Dict) -> Dict:
        move_i, cmd_i = divmod(idx, N_COMMANDS)
        deltas = self._team_deltas(TEAM_MOVES[move_i], state)
        command = COMMANDS[cmd_i]

        # The contract passes whenever commands[0] != current holder - there is
        # no "no pass" value. For none/shoot the pass target must be the holder
        # itself, or the ball gets passed back to the GK (idx 0) first.
        holder = (
            int(state["ball_holder"][1])
            if int(state["ball_holder"][0]) == self.player_team
            else 0
        )
        pass_target = (
            self._best_pass_target(state) if command == "pass" else holder
        )

        return {
            "player_deltas": deltas,
            "pass_target": pass_target,
            "shoot": 1 if command == "shoot" else 0,
        }

    def _team_deltas(self, move: str, state: Dict) -> List[List[int]]:
        if move == "hold":
            return [[0, 0]] * N_PLAYERS
        if move == "attack":
            return [[DELTA_X_RANGE, 0]] * N_PLAYERS
        if move == "defend":
            return [[-DELTA_X_RANGE, 0]] * N_PLAYERS
        if move == "push_up":
            return [[0, -DELTA_Y_RANGE]] * N_PLAYERS
        if move == "push_down":
            return [[0, DELTA_Y_RANGE]] * N_PLAYERS
        if move == "chase_ball":
            bh_team, bh_player = state["ball_holder"]
            bx, by = state["player_positions"][int(bh_team)][int(bh_player)]
            deltas = []
            for px, py in state["player_positions"][self.player_team]:
                deltas.append([
                    int(np.sign(bx - px)) * DELTA_X_RANGE,
                    int(np.sign(by - py)) * DELTA_Y_RANGE,
                ])
            return deltas
        return [[0, 0]] * N_PLAYERS

    def _best_pass_target(self, state: Dict) -> int:
        """Heuristic pass target: most advanced field player other than the
        holder (player 0 is the frozen GK - never worth passing to)."""
        positions = np.asarray(
            state["player_positions"][self.player_team], dtype=np.float32
        )
        xs = positions[:, 0].copy()
        xs[0] = -np.inf  # exclude goalkeeper
        bh_team, bh_player = int(state["ball_holder"][0]), int(state["ball_holder"][1])
        if bh_team == self.player_team and 0 <= bh_player < N_PLAYERS:
            xs[bh_player] = -np.inf
        # team 0 attacks +x, team 1 attacks -x
        if self.player_team == 0:
            return int(np.argmax(xs))
        return int(np.argmin(np.where(np.isfinite(xs), xs, np.inf)))

    def _action_to_index(self, action: Dict) -> int:
        """Map an env action dict back to its macro-action index."""
        if self._last_action is not None and id(action) == self._last_action[0]:
            return self._last_action[1]

        # Fallback classification for actions produced elsewhere
        shoot = int(action.get("shoot", 0))
        cmd_i = 2 if shoot else (1 if action.get("pass_target") else 0)

        deltas = np.asarray(action.get("player_deltas") or [[0, 0]] * N_PLAYERS, dtype=float)
        if not deltas.any():
            move_i = 0
        else:
            mean = deltas.mean(axis=0)
            if abs(mean[0]) >= abs(mean[1]):
                move_i = 1 if mean[0] > 0 else 2
            else:
                move_i = 3 if mean[1] < 0 else 4
        return move_i * N_COMMANDS + cmd_i

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, state: Dict, action: Dict, reward: float, next_state: Dict) -> None:
        """Store the transition and run a gradient step."""
        idx = self._action_to_index(action)
        s = encode_state(state, self.player_team)
        s_next = encode_state(next_state, self.player_team)
        done = next_state["state_idx"][0] >= N_ROUNDS
        self.replay.push(s, idx, float(reward), s_next, done)

        self.exploration_rate = max(
            MIN_EXPLORATION_RATE,
            self.exploration_rate * EXPLORATION_DECAY,
        )

        if len(self.replay) >= MIN_REPLAY_SIZE:
            self._update()

    def _update(self) -> None:
        s, a, r, s_next, done = self.replay.sample(BATCH_SIZE)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q = self.policy_net(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy net selects, target net evaluates
            next_a = self.policy_net(s_next).argmax(1, keepdim=True)
            next_q = self.target_net(s_next).gather(1, next_a).squeeze(1)
            target = r + self.discount_factor * next_q * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q, target)
        self.last_loss = float(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "exploration_rate": self.exploration_rate,
            "learn_steps": self.learn_steps,
        }, filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.exploration_rate = ckpt["exploration_rate"]
        self.learn_steps = ckpt["learn_steps"]
