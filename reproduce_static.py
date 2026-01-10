
import numpy as np
import mujoco
from cable_env import CableInsertionEnv
import time

env = CableInsertionEnv()
obs, _ = env.reset()

# Target joint position to align gripper (0.1) with socket (0.18)
# Joint x should be 0.08.
# Z should be such that connector enters socket.
# Gripper z=0.2, connector length=0.15 -> tip=0.05. Socket z=0.04.
# If we lower gripper by 0.0, tip is at 0.05. Socket is at 0.04.
# We might need to lower z slightly more or just be there.
# Let's try [0.08, 0, 0.0]

target_action = np.array([0.08, 0, 0.0])

print(f"Testing static reach to {target_action}...")

# Move slowly to avoid swing
current_action = np.array([0.0, 0, 0.0])
steps = 200

for i in range(steps):
    alpha = i / steps
    action = current_action * (1 - alpha) + target_action * alpha
    obs, reward, done, _, info = env.step(action)
    if i % 20 == 0:
        print(f"Step {i}: dist {info['distance']:.4f}, obs {obs[:3]}")

print(f"Final distance: {info['distance']:.4f}")
