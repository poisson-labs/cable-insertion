# test_reach.py
from cable_env import CableInsertionEnv
import numpy as np

env = CableInsertionEnv()
obs, _ = env.reset()

# Try a bunch of positions
best_dist = 1.0
best_action = None

for x in np.arange(0.1, 0.3, 0.02):
    for z in np.arange(0.0, 0.2, 0.02):
        env.reset()
        action = [x, 0, z]
        for _ in range(50):  # let it settle
            obs, _, _, _, info = env.step(action)
        if info['distance'] < best_dist:
            best_dist = info['distance']
            best_action = action
            print(f"New best: {best_dist:.4f} at {action}")

print(f"\nBest achievable: {best_dist:.4f} at {best_action}")