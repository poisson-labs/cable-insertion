import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.cable_env import CableInsertionEnv

env = CableInsertionEnv()
obs, _ = env.reset()
print(f"Initial obs: {obs}")

# Take a few random actions
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f"Step {i}: reward={reward:.3f}, dist={info['distance']:.3f}")
