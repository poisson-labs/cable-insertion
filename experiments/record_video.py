"""Record a tiled 3-camera video of a rollout.

Usage:
    python experiments/record_video.py                    # random policy
    python experiments/record_video.py --model models/cable_ppo   # trained policy
    python experiments/record_video.py --steps 300 --fps 15
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import imageio.v3 as iio
from envs.cable_env import CableInsertionEnv

CAMERAS = ["overhead", "side", "wrist"]
CAM_WIDTH = 320
CAM_HEIGHT = 240


def make_label(text, width):
    """Burn a text label into a small bar (pure numpy, no font deps)."""
    bar = np.zeros((16, width, 3), dtype=np.uint8)
    bar[:] = 30
    return bar


def tile_views(frames, dist):
    """Stack camera views side-by-side with a thin label bar on top."""
    labels = [make_label(cam, CAM_WIDTH) for cam in CAMERAS]
    top = np.concatenate(labels, axis=1)
    row = np.concatenate(frames, axis=1)
    return np.concatenate([top, row], axis=0)


def record(args):
    env = CableInsertionEnv(
        render_mode="rgb_array", render_width=CAM_WIDTH, render_height=CAM_HEIGHT
    )

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        print(f"Loaded policy: {args.model}")
    else:
        print("No --model given, using random policy")

    obs, _ = env.reset()
    frames = []
    episodes = 0

    for step in range(args.steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        dist = info["distance"]

        views = [env.render_camera(cam, CAM_WIDTH, CAM_HEIGHT) for cam in CAMERAS]
        frames.append(tile_views(views, dist))

        if done:
            episodes += 1
            print(f"  Step {step:3d}: SUCCESS (dist={dist:.4f}) — episode {episodes}")
            obs, _ = env.reset()
        elif truncated:
            print(f"  Step {step:3d}: timeout  (dist={dist:.4f})")
            obs, _ = env.reset()

    env.close()

    out_path = ROOT / "experiments" / args.output
    iio.imwrite(
        str(out_path), np.stack(frames), fps=args.fps, codec="libx264", pixelformat="yuv420p"
    )

    dur = len(frames) / args.fps
    print(f"\nSaved {dur:.1f}s video ({len(frames)} frames @ {args.fps}fps)")
    print(f"  {out_path}")
    print(f"  {episodes} successful insertions in {args.steps} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record tiled multi-camera rollout video")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to trained model (omit for random policy)"
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of env steps to record")
    parser.add_argument("--fps", type=int, default=10, help="Video framerate")
    parser.add_argument("--output", type=str, default="rollout.mp4", help="Output filename")
    record(parser.parse_args())
