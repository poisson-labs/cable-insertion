"""20k-step PPO smoke test on CableInsertionUR5eEnv (state + vision modes).

Validates the full training loop is end-to-end functional with:
  - 7-dim action space (6 joint vel + gripper)
  - 22-dim state obs / Dict vision obs
"""

import sys
import time
import resource
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from envs.cable_env_ur5e import CableInsertionUR5eEnv

TIMESTEPS = 20_000
logs_dir = ROOT / "logs"
logs_dir.mkdir(exist_ok=True)


def run_smoke(mode):
    print(f"\n{'=' * 60}")
    print(f"  UR5e PPO Smoke Test — {mode} mode — {TIMESTEPS:,} steps")
    print(f"{'=' * 60}")

    env = CableInsertionUR5eEnv(
        obs_mode=mode,
        randomize=True,
    )
    print(f"  Action space:  {env.action_space}")
    print(f"  Obs space:     {env.observation_space}")

    policy = "MultiInputPolicy" if mode == "vision" else "MlpPolicy"

    model = PPO(
        policy,
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
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"ur5e_{mode}_smoke")
    elapsed = time.time() - t0

    fps = TIMESTEPS / elapsed
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print(f"\n  --- Results ({mode}) ---")
    print(f"  Wall time:  {elapsed:.1f}s")
    print(f"  FPS:        {fps:.0f} steps/s")
    print(f"  Peak RSS:   {rss_mb:.0f} MB")

    env.close()
    return {"mode": mode, "elapsed": elapsed, "fps": fps, "rss_mb": rss_mb}


if __name__ == "__main__":
    results = []

    results.append(run_smoke("state"))
    results.append(run_smoke("vision"))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  {r['mode']:8s}  {r['elapsed']:6.1f}s  {r['fps']:6.0f} fps  {r['rss_mb']:.0f} MB RSS"
        )
