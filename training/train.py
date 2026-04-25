"""Workstream-A Session 4 PPO training run.

Trains CableInsertionEnv (state-mode floating gripper) for 4M timesteps.
Stable Baselines3 PPO with the same hyperparameters that produced the original
cable_ppo.zip — the only changes vs. the pre-Session-4 train.py are
observability (wandb, monitor, threshold-trigger-rate callback) and a
session-suffixed checkpoint name to avoid colliding with the legacy artifact.

Env, reward, action space, hyperparameters: unchanged.

Required environment variable on the training host:
    WANDB_API_KEY  (set on RunPod via dashboard env vars or `export` before run)

Optional:
    WANDB_RUN_NAME  (defaults to `restart_session4_<UTC_iso8601>`)
"""

import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from envs.cable_env import CableInsertionEnv

TIMESTEPS = 4_000_000
SEED = 42


class ThresholdInfoWrapper(gym.Wrapper):
    """Surface per-episode threshold crossings + min-distance via terminal info.

    Adds three keys to ``info`` on the terminal step (only):
        - ``thresholds_crossed``: sorted list of {5, 3, 2} the episode triggered
        - ``min_distance_m``: minimum connector-target distance during episode
        - ``is_success``: True iff the env terminated via insertion (dist <= 2cm)

    Read-only — does not change reward, action, observation, or stepping.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._min_dist = float("inf")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        d = info.get("distance")
        if d is not None and d < self._min_dist:
            self._min_dist = float(d)
        if terminated or truncated:
            tc = getattr(self.env.unwrapped, "thresholds_crossed", set())
            info["thresholds_crossed"] = sorted(tc)
            info["min_distance_m"] = self._min_dist
            info["is_success"] = bool(terminated)
        return obs, reward, terminated, truncated, info


class ThresholdMetricsCallback(BaseCallback):
    """Log rolling +10/+25/+100 trigger rates and success rate to wandb."""

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self.bonus5: deque[int] = deque(maxlen=window)
        self.bonus3: deque[int] = deque(maxlen=window)
        self.bonus2: deque[int] = deque(maxlen=window)
        self.success: deque[int] = deque(maxlen=window)
        self.min_dist: deque[float] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "thresholds_crossed" not in info:
                continue
            tc = set(info["thresholds_crossed"])
            self.bonus5.append(1 if 5 in tc else 0)
            self.bonus3.append(1 if 3 in tc else 0)
            self.bonus2.append(1 if 2 in tc else 0)
            self.success.append(1 if info.get("is_success") else 0)
            md = info.get("min_distance_m")
            if md is not None and np.isfinite(md):
                self.min_dist.append(float(md))
            wandb.log(
                {
                    "task/+10_rate": float(np.mean(self.bonus5)),
                    "task/+25_rate": float(np.mean(self.bonus3)),
                    "task/+100_rate": float(np.mean(self.bonus2)),
                    "task/success_rate": float(np.mean(self.success)),
                    "task/min_distance_mean": (
                        float(np.mean(self.min_dist)) if self.min_dist else float("nan")
                    ),
                    "task/episodes_in_window": len(self.success),
                },
                step=self.num_timesteps,
            )
        return True


models_dir = ROOT / "models"
logs_dir = ROOT / "logs"
models_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

run_name = os.environ.get(
    "WANDB_RUN_NAME",
    f"restart_session4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
)

wandb_run = wandb.init(
    project="cable-insertion",
    name=run_name,
    config={
        "policy": "MlpPolicy",
        "total_timesteps": TIMESTEPS,
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "env": "CableInsertionEnv",
        "obs_mode": "state",
        "randomize": True,
        "max_steps": 200,
        "seed": SEED,
        "session": "Workstream-A Session 4",
    },
    sync_tensorboard=True,
    monitor_gym=False,
)

base_env = CableInsertionEnv(randomize=True)
env = ThresholdInfoWrapper(base_env)
env = Monitor(
    env,
    info_keywords=("is_success", "min_distance_m"),
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    ent_coef=0.01,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    seed=SEED,
    tensorboard_log=str(logs_dir),
)

callbacks = CallbackList(
    [
        CheckpointCallback(
            save_freq=100_000,
            save_path=str(models_dir),
            name_prefix="cable_ppo_session4",
        ),
        WandbCallback(
            gradient_save_freq=0,
            model_save_path=str(models_dir / f"wandb_{run_name}"),
            verbose=1,
        ),
        ThresholdMetricsCallback(window=100),
    ]
)

print(f"{'=' * 60}")
print("  Cable Insertion PPO — Workstream-A Session 4")
print(f"{'=' * 60}")
print(f"  wandb run:    {run_name}")
print(f"  wandb URL:    {wandb_run.url}")
print(f"  timesteps:    {TIMESTEPS:,}")
print(f"  checkpoints:  {models_dir}/cable_ppo_session4_*_steps.zip")
print(f"  TB logs:      {logs_dir}")
print(f"{'=' * 60}", flush=True)

t0 = time.time()
model.learn(
    total_timesteps=TIMESTEPS,
    callback=callbacks,
    progress_bar=False,
    tb_log_name=run_name,
)
elapsed = time.time() - t0

final_path = models_dir / "cable_ppo_session4_final"
model.save(str(final_path))

print(f"\n{'=' * 60}")
print(f"  Done in {elapsed / 3600:.2f}h")
print(f"  Final checkpoint: {final_path}.zip")
print(f"  wandb URL: {wandb_run.url}")
print(f"{'=' * 60}")

wandb_run.finish()
