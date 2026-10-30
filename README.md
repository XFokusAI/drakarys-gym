# Drakarys RL Environment

Reinforcement learning environment for the Drakarys_GAME_0 smart contract.

The project is live at: https://xfokus.ai/projects/drakarys

![Drakarys Promo](./drakarys-game-0-overview.png)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install foundry toolkit (for anvil):

```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

## Usage

### Basic environment usage

```python
from drakarys_env import DrakarysEnv

env = DrakarysEnv()
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training an agent

```bash
# Tabular Q-learning (default)
python train_agent.py

# Deep Q-Network (requires torch) - dedicated script with checkpoints,
# periodic greedy evals, and metrics output
python train_dqn.py --episodes 500 --resume dqn_model.pt

# or via the generic trainer
python train_agent.py --agent dqn --episodes 1000 --model-path dqn_model.pt
```

The DQN agent (`dqn_agent.py`) is a Double DQN with experience replay and a
target network. The contract action space is too large to explore directly,
so the agent picks from 18 macro-actions: 6 team-level movement patterns
(hold / attack / defend / push_up / push_down / chase_ball) x 3 ball commands
(none / pass / shoot). Observations are encoded as a 93-dim normalized vector
relative to the agent's team.

### Custom agent implementation

```python
from drakarys_env import DrakarysEnv
from eth_account import Account

class MyAgent:
    def __init__(self, private_key=None):
        # Each agent has its own private key for signing commitments
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = Account.create()
        self.private_key = self.account.key.hex()
        self.address = self.account.address

    def get_action(self, state):
        return {
            "player_deltas": [[0, 0]] * 11,
            "pass_target": 0,
            "shoot": 0,
        }

# Create agent with its own key
agent = MyAgent()

# Environment signs with agent's key for its moves
env = DrakarysEnv(player_key=agent.private_key)
env.player_account = agent.account

state, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    action = agent.get_action(state)
    state, reward, done, truncated, info = env.step(action)

env.close()
```

## Environment Details

### Observation Space

- `player_positions`: (2, 11, 2) - x,y coordinates for each player
- `ball_holder`: (2,) - team_idx, player_idx of ball holder
- `score`: (2,) - goals for each team
- `state_idx`: (1,) - current round number
- `player_stats`: (2, 11, 2) - skill and speed for each player

### Action Space

- `player_deltas`: (11, 2) - x,y movement for each player
- `pass_target`: int - player index to pass to (0-10)
- `shoot`: int - whether to shoot (0 or 1)

### Game Rules

- 2 teams of 11 players each
- Field size: 1024x512
- 20 rounds per match
- Players can move (deltas), pass, and shoot
- Commit-reveal protocol for fairness

## Current Status

- ✅ Environment automatically starts local anvil chain
- ✅ Contract deployment works (deploys from compiled artifacts)
- ✅ Gym-compatible interface implemented
- ✅ Basic training loop with Q-learning agent
- ✅ Test script verifies environment functionality
- ⚠️ Contract calls execute but require match to exist on-chain
- ⚠️ State reading works but match creation needs proper on-chain init

## Next Steps

To fully integrate with real matches:

1. **Match creation**: The environment currently generates random match indices but doesn't create them on-chain. Implement match creation via contract call.

2. **Commit/reveal protocol**: The current implementation commits and reveals but doesn't wait for the contract to process them. Add proper phase tracking.

3. **State synchronization**: Ensure the environment reads the actual contract state after each reveal.

4. **Reward tuning**: Current reward is basic (goals only). Add rewards for:
   - Possession time
   - Field position
   - Successful passes
   - Interceptions
   - Time pressure (quicker wins = more reward)

5. **Multi-agent training**: Currently simulates opponent with random moves. Support training multiple agents together or against other agents.
