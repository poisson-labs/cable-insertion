"""4M-step PPO training on CableInsertionUR5eEnv (vision mode).

Usage:
    python training/train_ur5e_vision.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.cable_env_ur5e import CableInsertionUR5eEnv

TIMESTEPS = 4_000_000
models_dir = ROOT / "models"
logs_dir = ROOT / "logs"
models_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

env = CableInsertionUR5eEnv(obs_mode="vision", randomize=True)

print(f"{'='*60}")
print(f"  UR5e PPO Vision Training — {TIMESTEPS:,} steps")
print(f"{'='*60}")
print(f"  Action space:  {env.action_space}")
print(f"  Obs space:     {env.observation_space}")

checkpoint_cb = CheckpointCallback(
    save_freq=500_000,
    save_path=str(models_dir),
    name_prefix="ur5e_vision_ppo",
)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    ent_coef=0.01,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log=str(logs_dir),
)

t0 = time.time()
model.learn(
    total_timesteps=TIMESTEPS,
    callback=checkpoint_cb,
    tb_log_name="ur5e_vision_4M",
)
elapsed = time.time() - t0

model.save(str(models_dir / "ur5e_vision_4M"))

print(f"\n{'='*60}")
print(f"  Done in {elapsed/3600:.1f}h")
print(f"  Saved: {models_dir / 'ur5e_vision_4M.zip'}")
print(f"  Checkpoints: {models_dir / 'ur5e_vision_ppo_*_steps.zip'}")
print(f"{'='*60}")

env.close()
