import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from envs.cable_env import CableInsertionEnv

env = CableInsertionEnv(randomize=True)  # Match training config
model = PPO.load(str(ROOT / "models" / "cable_ppo"))

obs, _ = env.reset()
successes = 0
steps = 0
dists = []

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running() and steps < 5000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        viewer.sync()

        dists.append(info['distance'])
        steps += 1

        if done:
            successes += 1
            obs, _ = env.reset()

print(f"\n=== Results ({steps} steps) ===")
print(f"Successes: {successes}")
print(f"Min dist: {min(dists):.4f}")
print(f"Max dist: {max(dists):.4f}")
print(f"Mean dist: {np.mean(dists):.4f}")
print(f"Median dist: {np.median(dists):.4f}")
print(f"<2cm: {sum(1 for d in dists if d < 0.02)}")
print(f"<3cm: {sum(1 for d in dists if d < 0.03)}")
print(f"<5cm: {sum(1 for d in dists if d < 0.05)}")
