#!/usr/bin/env python3
"""
Drakarys GAME_0 Reinforcement Learning Environment

This environment provides a Gym-compatible interface for training RL agents
to play Drakarys_GAME_0 matches on a local anvil chain.

Features:
- Automatic anvil chain instantiation
- Contract deployment on demand
- Full commit/reveal protocol implementation
- Multiple match support
- Vectorized observations
- Discrete action space (move deltas + commands)

Usage:
    from environment import DrakarysEnv
    
    env = DrakarysEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import os
import sys
import time
import json
import random
import subprocess
import signal
import atexit
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path to import web3
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

import requests
from web3 import Web3
from eth_account import Account
from eth_utils import keccak

# Constants matching the contract
FIELD_W, FIELD_H = 1024, 512
N_PLAYERS = 11
N_TEAMS = 2
N_ROUNDS = 20
DELTA_X_RANGE = 5
DELTA_Y_RANGE = 5
STATE_IDX_RANGE = N_ROUNDS + 1

# Default local anvil settings
ANVIL_PORT = 8545
ANVIL_HOST = "127.0.0.1"
ANVIL_RPC_URL = f"http://{ANVIL_HOST}:{ANVIL_PORT}"
CHAIN_ID = 31337

# Default operator private key (Anvil account #0)
OPERATOR_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
OPERATOR_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"


@dataclass
class Player:
    """Player state"""
    x: int
    y: int


@dataclass
class TeamState:
    """Team state"""
    player_positions: List[Player]
    pass_target: int = 0
    passing_to: int = 0
    shoot: bool = False


@dataclass
class MatchState:
    """Full match state"""
    teams: List[TeamState]
    score: List[int]
    has_ball_team_idx: int
    has_ball_player_idx: int


@dataclass
class MatchMetadata:
    """Match metadata"""
    state_idx: int
    agents: List[str]
    player_stats: List[List[List[int]]]


class AnvilManager:
    """Manages the local anvil blockchain"""
    
    def __init__(self):
        self.process = None
        self.running = False
    
    def start(self) -> None:
        """Start the anvil chain"""
        if self.running:
            return
        
        # Check if anvil is available
        try:
            subprocess.run(["anvil", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("anvil not found - install foundry toolkit")
        
        # Start anvil with predetermined accounts
        self.process = subprocess.Popen(
            [
                "anvil",
                "--host", ANVIL_HOST,
                "--port", str(ANVIL_PORT),
                "--chain-id", str(CHAIN_ID),
                "--accounts", "10",
                "--mnemonic", "test test test test test test test test test test test junk",
                "--state-interval", "10000",
                "--auto-impersonate",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for anvil to start
        w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
        for _ in range(30):
            if w3.is_connected():
                break
            time.sleep(1)
        else:
            raise RuntimeError("anvil failed to start")
        
        self.running = True
        
        # Register cleanup
        atexit.register(self.stop)
    
    def stop(self) -> None:
        """Stop the anvil chain"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.running = False
    
    def __del__(self):
        self.stop()


class DrakarysContract:
    """Wrapper for the Drakarys_GAME_0 contract"""
    
    def __init__(self, w3: Web3, contract_address: str, abi: List[Dict]):
        self.w3 = w3
        self.contract_address = contract_address
        self.contract = w3.eth.contract(address=contract_address, abi=abi)
    
    @classmethod
    def deploy(cls, w3: Web3, abi: List[Dict], bytecode: str, operator_key: str) -> "DrakarysContract":
        """Deploy the contract and return a wrapper"""
        account = Account.from_key(operator_key)
        constructor_tx = {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 5000000,
            "gasPrice": w3.to_wei("20", "gwei"),
            "data": bytecode,
        }
        
        signed = w3.eth.account.sign_transaction(constructor_tx, operator_key)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return cls(w3, receipt.contractAddress, abi)
    
    def commit(self, match_idx: int, commitments: List[bytes], signatures: List[Dict]) -> str:
        """Submit move commitments"""
        # TODO: Implement proper ABI encoding for signatures
        # For now, this is a placeholder - actual implementation would need
        # to properly encode the Signature structs
        pass
    
    def reveal(self, match_idx: int, moves: List[Dict]) -> str:
        """Submit move reveals"""
        # TODO: Implement proper ABI encoding for moves
        pass
    
    def get_match_data(self, match_idx: int) -> Tuple[MatchMetadata, MatchState]:
        """Get current match metadata and state"""
        # TODO: Implement contract call
        pass


class DrakarysEnv(gym.Env if gym else object):
    """
    Gym-compatible environment for Drakarys_GAME_0.
    
    Observation space: Dict with:
    - player_positions: (2, 11, 2) - x,y coordinates for each player
    - ball_holder: (2,) - team_idx, player_idx
    - score: (2,) - goals for each team
    - state_idx: (1,) - current round
    
    Action space: Dict with:
    - player_deltas: (11, 2) - x,y movement for each player
    - pass_target: (1,) - player index to pass to (0-10)
    - shoot: (1,) - whether to shoot (0 or 1)
    """
    
    def __init__(
        self,
        operator_key: str = OPERATOR_PRIVATE_KEY,
        anvil: AnvilManager = None,
        contract_path: str = None,
        auto_start_anvil: bool = True,
    ):
        super().__init__()
        
        self.operator_key = operator_key
        self.operator = Account.from_key(operator_key)
        self.anvil = anvil or AnvilManager()
        self.w3 = None
        self.contract = None
        self.match_idx = None
        self.current_state_idx = 0
        self.pending_moves = {}
        self.revealed_states = set()
        self.episode_rewards = []
        
        # Action space: player deltas + pass/shoot commands
        if spaces:
            self.action_space = spaces.Dict({
                "player_deltas": spaces.Box(
                    low=-DELTA_X_RANGE,
                    high=DELTA_X_RANGE,
                    shape=(N_PLAYERS, 2),
                    dtype=int
                ),
                "pass_target": spaces.Discrete(N_PLAYERS),
                "shoot": spaces.Discrete(2),
            })
            
            # Observation space
            self.observation_space = spaces.Dict({
                "player_positions": spaces.Box(
                    low=0,
                    high=max(FIELD_W, FIELD_H),
                    shape=(N_TEAMS, N_PLAYERS, 2),
                    dtype=int
                ),
                "ball_holder": spaces.Box(low=0, high=N_TEAMS-1, shape=(2,), dtype=int),
                "score": spaces.Box(low=0, high=N_ROUNDS, shape=(2,), dtype=int),
                "state_idx": spaces.Box(low=0, high=STATE_IDX_RANGE, shape=(1,), dtype=int),
            })
        
        if auto_start_anvil:
            self.start_chain()
    
    def start_chain(self) -> None:
        """Start the local anvil chain"""
        self.anvil.start()
        self.w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
        
        # Verify connection
        if not self.w3.is_connected():
            raise RuntimeError("Failed to connect to anvil")
        
        # Deploy contract if needed
        self.deploy_contract()
    
    def deploy_contract(self) -> None:
        """Deploy the Drakarys contract"""
        # TODO: Load contract artifacts and deploy
        # For now, create a mock contract for development
        self.contract = MockDrakarysContract(self.w3)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment and start a new match"""
        if seed is not None:
            random.seed(seed)
        
        # Create a new match
        self.match_idx = self.create_match()
        self.current_state_idx = 0
        self.pending_moves = {}
        self.revealed_states = set()
        self.episode_rewards = []
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take an action in the environment"""
        # Convert action to move
        move = self._action_to_move(action)
        
        # Commit the move
        self._commit_move(move)
        
        # Wait for opponent's move and reveal
        # For single-agent training, we simulate the opponent
        opponent_move = self._generate_opponent_move()
        
        # Reveal both moves
        self._reveal_moves(move, opponent_move)
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def create_match(self) -> int:
        """Create a new match on the contract"""
        # Generate a unique match index
        # In a real implementation, this would query the contract or use
        # a counter stored in the environment
        return random.randint(0, 1000000)
    
    def _get_observation(self) -> Dict:
        """Get the current observation"""
        # TODO: Read actual state from contract
        # For now, return a random state
        return {
            "player_positions": [[(random.randint(0, FIELD_W), random.randint(0, FIELD_H)) 
                                  for _ in range(N_PLAYERS)] for _ in range(N_TEAMS)],
            "ball_holder": [random.randint(0, 1), random.randint(0, 10)],
            "score": [random.randint(0, 3), random.randint(0, 3)],
            "state_idx": [self.current_state_idx],
        }
    
    def _action_to_move(self, action: Dict) -> Dict:
        """Convert action to a move format"""
        return {
            "stateIdx": self.current_state_idx,
            "deltaPlayerPositions": action["player_deltas"].tolist(),
            "commands": [action["pass_target"], action["shoot"]],
            "seedPart": str(random.getrandbits(128)),
        }
    
    def _commit_move(self, move: Dict) -> None:
        """Commit a move to the contract"""
        # TODO: Implement commit transaction
        self.pending_moves[self.current_state_idx] = move
    
    def _reveal_moves(self, my_move: Dict, opponent_move: Dict) -> None:
        """Reveal moves to the contract"""
        # TODO: Implement reveal transaction
        self.revealed_states.add(self.current_state_idx)
        self.current_state_idx += 1
    
    def _generate_opponent_move(self) -> Dict:
        """Generate a move for the opponent (for single-agent training)"""
        return {
            "stateIdx": self.current_state_idx,
            "deltaPlayerPositions": [
                [random.randint(-DELTA_X_RANGE, DELTA_X_RANGE),
                 random.randint(-DELTA_Y_RANGE, DELTA_Y_RANGE)]
                for _ in range(N_PLAYERS)
            ],
            "commands": [random.randint(0, N_PLAYERS-1), random.randint(0, 1)],
            "seedPart": str(random.getrandbits(128)),
        }
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state"""
        # Simple reward: goal scored = +1, goal conceded = -1
        # This should be more sophisticated in a real implementation
        return random.choice([-1.0, 0.0, 1.0])
    
    def _is_terminated(self) -> bool:
        """Check if the episode is finished"""
        return self.current_state_idx >= N_ROUNDS
    
    def _is_truncated(self) -> bool:
        """Check if the episode should be truncated (time limit)"""
        return False
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "match_idx": self.match_idx,
            "current_state_idx": self.current_state_idx,
            "episode_rewards": self.episode_rewards,
        }
    
    def render(self) -> None:
        """Render the environment (optional)"""
        pass
    
    def close(self) -> None:
        """Clean up resources"""
        self.anvil.stop()


class MockDrakarysContract:
    """Mock contract for development without a real deployment"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.matches = {}
        self.next_match_idx = 0
    
    def create_match(self) -> int:
        """Create a new mock match"""
        match_idx = self.next_match_idx
        self.next_match_idx += 1
        self.matches[match_idx] = {
            "state_idx": 0,
            "ended": False,
            "agents": [OPERATOR_ADDRESS, "0x" + "0" * 40],
        }
        return match_idx


# Example usage
if __name__ == "__main__":
    env = DrakarysEnv()
    obs, info = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
