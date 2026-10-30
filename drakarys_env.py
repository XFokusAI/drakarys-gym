#!/usr/bin/env python3
"""
Drakarys GAME_0 Reinforcement Learning Environment

Complete implementation for training RL agents on local anvil chain.
Handles contract deployment, match creation, and full game protocol.
"""

import os
import sys
import time
import json
import random
import subprocess
import atexit
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

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

# Game constants
FIELD_W, FIELD_H = 1024, 512
N_PLAYERS = 11
N_TEAMS = 2
N_ROUNDS = 20
DELTA_X_RANGE = 5
DELTA_Y_RANGE = 5

# Anvil settings
ANVIL_PORT = 8545
ANVIL_HOST = "127.0.0.1"
ANVIL_RPC_URL = f"http://{ANVIL_HOST}:{ANVIL_PORT}"
CHAIN_ID = 31337

# Default anvil accounts
OPERATOR_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
OPERATOR_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

# Contract paths
PROJECT_ROOT = Path(__file__).parent.parent
CONTRACT_JSON_PATH = PROJECT_ROOT / "sc" / "out" / "Drakarys_GAME_0.sol" / "Drakarys_GAME_0.json"


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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
    def deploy(cls, w3: Web3, contract_artifact: Dict, operator_key: str) -> "DrakarysContract":
        """Deploy the contract and return a wrapper"""
        account = Account.from_key(operator_key)
        
        # Extract bytecode and ABI from artifact
        bytecode = contract_artifact["bytecode"]["object"]
        abi = contract_artifact["abi"]
        
        # Deploy contract
        contract = w3.eth.contract(abi=abi, bytecode=bytecode)
        
        constructor_tx = contract.constructor().build_transaction({
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 5000000,
            "gasPrice": w3.to_wei("20", "gwei"),
        })
        
        signed = w3.eth.account.sign_transaction(constructor_tx, operator_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return cls(w3, receipt.contractAddress, abi)
    
    def commit(self, match_idx: int, commitments: List[bytes], signatures: List[Dict], sender_key: str) -> str:
        """Submit move commitments"""
        account = Account.from_key(sender_key)
        
        # Encode signatures as tuples - convert bytes to proper format
        sig_tuples = []
        for sig in signatures:
            # Ensure r and s are bytes32
            r = sig["r"] if isinstance(sig["r"], bytes) else bytes.fromhex(sig["r"].replace("0x", ""))
            s = sig["s"] if isinstance(sig["s"], bytes) else bytes.fromhex(sig["s"].replace("0x", ""))
            sig_tuples.append((int(sig["v"]), r, s))
        
        tx = self.contract.functions.commit(match_idx, commitments, sig_tuples).build_transaction({
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address, "pending"),
            "gas": 500000,
            "gasPrice": self.w3.to_wei("20", "gwei"),
        })
        
        signed = self.w3.eth.account.sign_transaction(tx, sender_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def reveal(self, match_idx: int, moves: List[Dict], sender_key: str) -> str:
        """Submit move reveals"""
        account = Account.from_key(sender_key)
        
        # Convert moves to contract format - match the exact struct structure
        move_tuples = []
        for move in moves:
            # Ensure deltaPlayerPositions is a list of lists of ints
            deltas = [[int(x), int(y)] for x, y in move["deltaPlayerPositions"]]
            # Ensure commands is a list of ints
            commands = [int(c) for c in move["commands"]]
            move_tuples.append((
                int(move["stateIdx"]),
                deltas,
                commands,
                int(move["seedPart"])
            ))
        
        fn = self.contract.functions.reveal(match_idx, move_tuples)
        # init reveal writes ~90 fresh storage slots (~1.5M gas) - estimate
        # with margin instead of the old fixed 500k, which silently reverted
        gas = int(fn.estimate_gas({"from": account.address}) * 1.5)
        tx = fn.build_transaction({
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address, "pending"),
            "gas": gas,
            "gasPrice": self.w3.to_wei("20", "gwei"),
        })
        
        signed = self.w3.eth.account.sign_transaction(tx, sender_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def get_match_data(self, match_idx: int):
        """Get current match metadata and state"""
        try:
            result = self.contract.functions.getMatchData(match_idx).call()
            # Result is a tuple: (metadata, state)
            # metadata = (stateIdx, agents, playerStats)
            # state = (teams, score, hasBallTeamIdx, hasBallPlayerIdx)
            metadata, state = result
            
            # Convert tuples to dict format for easier use
            metadata_dict = {
                "stateIdx": metadata[0],
                "agents": list(metadata[1]),
                "playerStats": [list(team) for team in metadata[2]],
            }
            
            # teams is a tuple of tuples: (team0, team1)
            # each team = (playerPositions, pass, passingTo, shoot)
            teams_list = []
            for team in state[0]:
                teams_list.append({
                    "playerPositions": [list(p) for p in team[0]],
                    "pass": team[1],
                    "passingTo": team[2],
                    "shoot": team[3],
                })
            
            state_dict = {
                "teams": teams_list,
                "score": list(state[1]),
                "hasBallTeamIdx": state[2],
                "hasBallPlayerIdx": state[3],
            }
            
            return metadata_dict, state_dict
        except Exception as e:
            print(f"Error getting match data: {e}")
            return None, None
    
    def get_past_events(self, event_name: str, from_block: int = 0):
        """Get past events"""
        event = getattr(self.contract.events, event_name)
        # Use newer web3.py API
        return event.get_logs(from_block=from_block)


class DrakarysEnv(gym.Env if gym else object):
    """
    Gym-compatible environment for Drakarys_GAME_0.
    
    Observation space:
    - player_positions: (2, 11, 2) - x,y coordinates for each player
    - ball_holder: (2,) - team_idx, player_idx
    - score: (2,) - goals for each team
    - state_idx: (1,) - current round
    - player_stats: (2, 11, 2) - skill and speed for each player
    
    Action space:
    - player_deltas: (11, 2) - x,y movement for each player
    - pass_target: (1,) - player index to pass to (0-10)
    - shoot: (1,) - whether to shoot (0 or 1)
    """
    
    def __init__(
        self,
        operator_key: str = OPERATOR_PRIVATE_KEY,
        player_key: str = None,  # Agent's private key for signing
        contract_path: str = None,
        auto_start_anvil: bool = True,
        player_team: int = 0,  # 0 or 1 - which team the agent controls
    ):
        super().__init__()
        
        self.operator_key = operator_key
        self.operator = Account.from_key(operator_key)
        
        # Agent's account for signing moves
        if player_key:
            self.player_account = Account.from_key(player_key)
        else:
            self.player_account = Account.create()
        self.player_key = self.player_account.key.hex()
        self.player_address = self.player_account.address
        
        self.player_team = player_team
        self.anvil = AnvilManager()
        self.w3 = None
        self.contract = None
        self.match_idx = None
        self.current_state_idx = 0
        self.pending_moves = {}
        self.revealed_states = set()
        self.episode_rewards = []
        self.last_score = [0, 0]
        self.contract_artifact = None
        
        # Load contract artifact
        if contract_path and Path(contract_path).exists():
            with open(contract_path, 'r') as f:
                self.contract_artifact = json.load(f)
        else:
            # Try default path
            if CONTRACT_JSON_PATH.exists():
                with open(CONTRACT_JSON_PATH, 'r') as f:
                    self.contract_artifact = json.load(f)
        
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
                "state_idx": spaces.Box(low=0, high=N_ROUNDS, shape=(1,), dtype=int),
                "player_stats": spaces.Box(
                    low=0, high=99,
                    shape=(N_TEAMS, N_PLAYERS, 2),
                    dtype=int
                ),
            })
        
        if auto_start_anvil:
            self.start_chain()
    
    def start_chain(self) -> None:
        """Start the local anvil chain"""
        self.anvil.start()
        self.w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
        
        if not self.w3.is_connected():
            raise RuntimeError("Failed to connect to anvil")
        
        # Deploy contract
        self.deploy_contract()
    
    def deploy_contract(self) -> None:
        """Deploy the Drakarys contract"""
        if self.contract_artifact:
            self.contract = DrakarysContract.deploy(
                self.w3, self.contract_artifact, self.operator_key
            )
            print(f"Contract deployed at {self.contract.contract_address}")
        else:
            print("Warning: No contract artifact found - using mock mode")
            self.contract = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment and start a new match"""
        if seed is not None:
            random.seed(seed)
        
        # Create a new match
        self.match_idx = self._create_match()
        self.current_state_idx = 0
        self.pending_moves = {}
        self.revealed_states = set()
        self.episode_rewards = []
        self.last_score = [0, 0]
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take an action in the environment"""
        # Convert action to move
        move = self._action_to_move(action)
        
        # Simulate opponent move
        opponent_move = self._generate_opponent_move()
        
        # Execute both moves (commit + reveal)
        self._execute_round(move, opponent_move)
        
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
    
    def _create_match(self) -> int:
        """Create a new match"""
        # Generate a unique match index
        match_idx = random.randint(0, 1000000)
        
        if self.contract is not None:
            try:
                # Try to get past MatchStarted events to find the next available index
                events = self.contract.get_past_events("MatchStarted", from_block=0)
                if events:
                    used_indices = {e["args"]["matchIdx"] for e in events}
                    match_idx = max(used_indices) + 1 if used_indices else 0
            except Exception as e:
                print(f"Could not query past events: {e}")
        
        return match_idx
    
    def _get_observation(self) -> Dict:
        """Get the current observation"""
        if self.contract is None:
            # Mock observation
            return {
                "player_positions": [
                    [[random.randint(0, FIELD_W), random.randint(0, FIELD_H)] for _ in range(N_PLAYERS)]
                    for _ in range(N_TEAMS)
                ],
                "ball_holder": [random.randint(0, 1), random.randint(0, 10)],
                "score": [random.randint(0, 3), random.randint(0, 3)],
                "state_idx": [self.current_state_idx],
                "player_stats": [
                    [[random.randint(50, 99), random.randint(50, 99)] for _ in range(N_PLAYERS)]
                    for _ in range(N_TEAMS)
                ],
            }
        
        # Read real state from contract
        try:
            metadata, state = self.contract.get_match_data(self.match_idx)
            if metadata and state:
                # Convert contract data to observation format
                # Contract returns tuples, convert to lists for JSON compatibility
                return {
                    "player_positions": [
                        [list(p) for p in team["playerPositions"]]
                        for team in state["teams"]
                    ],
                    "ball_holder": [state["hasBallTeamIdx"], state["hasBallPlayerIdx"]],
                    "score": list(state["score"]),
                    "state_idx": [metadata["stateIdx"]],
                    "player_stats": [
                        [list(p) for p in team]
                        for team in metadata["playerStats"]
                    ],
                }
        except Exception as e:
            print(f"Error reading contract state: {e}")
            # Return a default observation on error
            return {
                "player_positions": [[[0, 0] for _ in range(N_PLAYERS)] for _ in range(N_TEAMS)],
                "ball_holder": [0, 0],
                "score": [0, 0],
                "state_idx": [self.current_state_idx],
                "player_stats": [[[0, 0] for _ in range(N_PLAYERS)] for _ in range(N_TEAMS)],
            }
        
        # Fallback - should not reach here
        return {
            "player_positions": [[[0, 0] for _ in range(N_PLAYERS)] for _ in range(N_TEAMS)],
            "ball_holder": [0, 0],
            "score": [0, 0],
            "state_idx": [self.current_state_idx],
            "player_stats": [[[0, 0] for _ in range(N_PLAYERS)] for _ in range(N_TEAMS)],
        }
    
    def _action_to_move(self, action: Dict) -> Dict:
        """Convert action to a move format"""
        # Ensure all values are native Python ints (not numpy types)
        deltas = [[int(x), int(y)] for x, y in action["player_deltas"]]
        return {
            "stateIdx": int(self.current_state_idx),
            "deltaPlayerPositions": deltas,
            "commands": [int(action["pass_target"]), int(action["shoot"])],
            "seedPart": str(random.getrandbits(128)),
        }
    
    def _generate_opponent_move(self) -> Dict:
        """Generate a move for the opponent"""
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
    
    def _execute_round(self, my_move: Dict, opponent_move: Dict) -> None:
        """Execute a round of the game"""
        if self.contract is None:
            # Mock execution
            self.current_state_idx += 1
            return
        
        try:
            # Commit phase - sign with the respective agent's key
            my_commit = self._create_commitment(my_move)
            opp_commit = self._create_commitment(opponent_move)
            
            # Agent signs with their own key, opponent uses operator key
            my_sig = self._sign_commitment(my_commit, self.player_account)
            opp_sig = self._sign_commitment(opp_commit, self.operator)
            
            sigs = [my_sig, opp_sig]
            
            # Operator submits both commitments
            rc = self.contract.commit(self.match_idx, [my_commit, opp_commit], sigs, self.operator_key)
            if getattr(rc, "status", 1) != 1:
                print(f"commit reverted (match {self.match_idx})")
                return

            # Reveal phase - operator reveals for both agents
            rr = self.contract.reveal(self.match_idx, [my_move, opponent_move], self.operator_key)
            if getattr(rr, "status", 1) != 1:
                print(f"reveal reverted (match {self.match_idx}, round {self.current_state_idx})")
                return

            self.revealed_states.add(self.current_state_idx)
            self.current_state_idx += 1
            
        except Exception as e:
            print(f"Error executing round: {e}")
    
    def _create_commitment(self, move: Dict) -> bytes:
        """Create a commitment hash for a move"""
        if move["stateIdx"] == 0:
            # init round commits the seed only
            return keccak(pack_u(self.match_idx) + pack_u(int(move["seedPart"])))
        
        data = pack_u(self.match_idx) + pack_u(move["stateIdx"])
        for delta in move["deltaPlayerPositions"]:
            data += pack_i(delta[0]) + pack_i(delta[1])
        data += pack_u(move["commands"][0]) + pack_u(move["commands"][1])
        data += pack_u(int(move["seedPart"]))
        return keccak(data)
    
    def _sign_commitment(self, commitment: bytes, account) -> Dict:
        """Sign a commitment hash with an account's private key"""
        signed = account.unsafe_sign_hash(commitment)
        # Convert r and s to bytes32 format
        r_bytes = int(signed.r).to_bytes(32, "big")
        s_bytes = int(signed.s).to_bytes(32, "big")
        return {
            "v": signed.v,
            "r": r_bytes,
            "s": s_bytes,
        }
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state"""
        try:
            obs = self._get_observation()
            score = obs["score"]
            
            # Reward for scoring a goal
            my_score = score[self.player_team]
            opp_score = score[1 - self.player_team]
            
            # Check if score changed
            score_diff = my_score - self.last_score[self.player_team]
            opp_score_diff = opp_score - self.last_score[1 - self.player_team]
            
            self.last_score = score.copy()
            
            reward = 0.0
            
            # +1 for scoring, -1 for conceding
            if score_diff > 0:
                reward += 1.0
            if opp_score_diff > 0:
                reward -= 1.0
            
            # Bonus for winning the match
            if self.current_state_idx >= N_ROUNDS:
                if my_score > opp_score:
                    reward += 5.0
                elif my_score < opp_score:
                    reward -= 5.0
            
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def _is_terminated(self) -> bool:
        """Check if the episode is finished"""
        return self.current_state_idx >= N_ROUNDS
    
    def _is_truncated(self) -> bool:
        """Check if the episode should be truncated"""
        return False
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "match_idx": self.match_idx,
            "current_state_idx": self.current_state_idx,
            "episode_rewards": self.episode_rewards,
        }
    
    def render(self) -> None:
        """Render the environment"""
        pass
    
    def close(self) -> None:
        """Clean up resources"""
        self.anvil.stop()


def pack_u(value: int) -> bytes:
    """Pack uint256"""
    return int(value).to_bytes(32, "big")


def pack_i(value: int) -> bytes:
    """Pack int256 (two's complement)"""
    return (int(value) % (1 << 256)).to_bytes(32, "big")


# Example usage
if __name__ == "__main__":
    env = DrakarysEnv()
    obs, info = env.reset()
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {info['current_state_idx']}: reward={reward}")
        
        if terminated or truncated:
            break
    
    env.close()
