import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from cable_env import CableInsertionEnv

env = CableInsertionEnv()
model = PPO.load("cable_ppo")

obs, _ = env.reset()
successes = 0
steps = 0
dists = []

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running() and steps < 5000:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
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
print(f"<3cm: {sum(1 for d in dists if d < 0.03)}")
print(f"<5cm: {sum(1 for d in dists if d < 0.05)}")
print(f"<10cm: {sum(1 for d in dists if d < 0.10)}")