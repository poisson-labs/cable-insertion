import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Compatibility shim for checkpoints saved with NumPy 2.x when loaded under NumPy 1.x
if not hasattr(np, "_core"):
    _core = types.ModuleType("numpy._core")
    _core.numeric = np.core.numeric
    _core.multiarray = np.core.multiarray
    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

from stable_baselines3 import PPO

from envs.cable_env import CableInsertionEnv

env = CableInsertionEnv(randomize=True)  # Match training config

# Resolve checkpoint path (root or models/)
model_path = ROOT / "cable_ppo.zip"
if not model_path.exists():
    model_path = ROOT / "models" / "cable_ppo.zip"
if not model_path.exists():
    model_path = ROOT / "models" / "cable_ppo"

print(f"Loading checkpoint from: {model_path}")
model = PPO.load(str(model_path))

obs, _ = env.reset(seed=42)
successes = 0
steps = 0
dists = []

# Try interactive viewer if available, fallback to headless evaluation
try:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            viewer.sync()

            dists.append(info["distance"])
            steps += 1

            if done or truncated:
                if done:
                    successes += 1
                obs, _ = env.reset()
except Exception as e:
    print(f"Running headless evaluation ({e})...")
    for ep in range(50):
        obs, _ = env.reset()
        for s in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            dists.append(info["distance"])
            steps += 1
            if done or truncated:
                if done:
                    successes += 1
                break

print(f"\n=== Results ({steps} steps, {successes} successes) ===")
print(f"Success rate: {successes / 50 * 100:.1f}%")
print(f"Min dist: {min(dists):.4f}m")
print(f"Max dist: {max(dists):.4f}m")
print(f"Mean dist: {np.mean(dists):.4f}m")
print(f"Median dist: {np.median(dists):.4f}m")
print(f"<2cm: {sum(1 for d in dists if d < 0.02)}")
print(f"<3cm: {sum(1 for d in dists if d < 0.03)}")
print(f"<5cm: {sum(1 for d in dists if d < 0.05)}")
