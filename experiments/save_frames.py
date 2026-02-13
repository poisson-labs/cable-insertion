"""Run a random policy for 10 steps and save rendered frames from all cameras."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from envs.cable_env import CableInsertionEnv

CAMERAS = ["overhead", "side", "wrist"]

output_dir = ROOT / "experiments" / "frames"
output_dir.mkdir(exist_ok=True)

env = CableInsertionEnv(render_mode="rgb_array")
obs, _ = env.reset()

for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    for cam in CAMERAS:
        frame = env.render_camera(cam)
        Image.fromarray(frame).save(output_dir / f"step_{i:02d}_{cam}.png")

    print(f"Step {i:2d}: dist={info['distance']:.4f}  saved {len(CAMERAS)} views ({frame.shape[1]}x{frame.shape[0]})")

    if done:
        obs, _ = env.reset()

env.close()
print(f"\nSaved {10 * len(CAMERAS)} frames to {output_dir}/")
