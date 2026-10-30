#!/usr/bin/env python3
"""
Test script for Drakarys RL environment.
Verifies basic functionality without running a full training loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from drakarys_env import DrakarysEnv, OPERATOR_PRIVATE_KEY


def test_environment():
    """Test basic environment functionality"""
    print("Testing Drakarys RL environment...")
    
    try:
        # Initialize environment
        env = DrakarysEnv(operator_key=OPERATOR_PRIVATE_KEY)
        print("✓ Environment initialized")
        print(f"  Contract address: {env.contract.contract_address if env.contract else 'None'}")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset")
        print(f"  Match index: {info['match_idx']}")
        print(f"  State index: {obs['state_idx'][0]}")
        
        # Test step
        action = env.action_space.sample()
        print("✓ Action sampled")
        print(f"  Action keys: {list(action.keys())}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Step executed")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  New state index: {obs['state_idx'][0]}")
        
        # Test multiple steps
        print("\nRunning 5 steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: state_idx={obs['state_idx'][0]}, reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
