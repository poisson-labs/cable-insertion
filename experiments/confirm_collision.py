import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from envs.cable_env import CableInsertionEnv

env = CableInsertionEnv()
obs, _ = env.reset()

# Lift, then move over, then drop.
# Waypoints (action coordinates):
waypoints = [
    np.array([0.0, 0, 0.1]),   # Lift gripper to z=+0.1 (abs 0.3)
    np.array([0.08, 0, 0.1]),  # Move over socket (abs x=0.18)
    np.array([0.08, 0, 0.0])   # Drop down
]

print("Testing 'Lift -> Move -> Drop' trajectory...")

for wp_idx, wp in enumerate(waypoints):
    print(f"Moving to waypoint {wp_idx}: {wp}")
    for _ in range(100):
        # Simple P control in action space
        # But here we just set the action to the waypoint because action IS position control
        obs, reward, done, _, info = env.step(wp)

    print(f"Reached WP {wp_idx}. Dist: {info['distance']:.4f}, Connector Z: {obs[2]:.4f}")

print(f"Final distance: {info['distance']:.4f}")
