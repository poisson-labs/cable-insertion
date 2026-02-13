"""Visualize UR5e cable insertion env: interactive viewer + camera montage.

Usage:
    python experiments/visualize_ur5e.py            # montage only
    python experiments/visualize_ur5e.py --viewer   # montage + interactive 3D viewer
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
from PIL import Image, ImageDraw
from envs.cable_env_ur5e import CableInsertionUR5eEnv

MONTAGE_CAMERAS = ["overhead", "wrist_left", "wrist_center", "wrist_right"]
MONTAGE_STEPS = 5
TILE_W, TILE_H = 320, 240
STEPS_BETWEEN_CAPTURES = 8  # run this many steps between each montage row
FRAMES_DIR = ROOT / "experiments" / "frames"
LABEL_H = 28  # height of label bar


def add_label(frame, text):
    """Burn a text label into the top of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # Semi-transparent bar at top
    draw.rectangle([0, 0, img.width, LABEL_H], fill=(0, 0, 0, 200))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.array(img)


def make_montage(env):
    """Scripted motion sequence — captures 4 camera views at 5 keyframes."""
    # Predefined actions that produce visible arm + gripper motion
    scripts = [
        {"joints": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0], "grip": -1.0},   # shoulder pan
        {"joints": [0.0, 0.4, -0.3, 0.0, 0.0, 0.0], "grip": -1.0},   # shoulder lift + elbow
        {"joints": [-0.3, 0.0, 0.0, 0.5, 0.3, 0.0], "grip": 1.0},    # wrist motion + close grip
        {"joints": [0.0, -0.3, 0.4, 0.0, 0.0, 0.5], "grip": 1.0},    # elbow + wrist roll
        {"joints": [-0.4, 0.2, -0.2, -0.3, 0.0, 0.0], "grip": -0.5}, # return + open grip
    ]

    rows = []
    for i, script in enumerate(scripts):
        action = np.zeros(7, dtype=np.float32)
        action[:6] = script["joints"]
        action[6] = script["grip"]

        # Run several steps so motion is visible
        for _ in range(STEPS_BETWEEN_CAPTURES):
            env.step(action)

        # Capture all cameras for this keyframe
        tiles = []
        for cam in MONTAGE_CAMERAS:
            frame = env.render_camera(cam, width=TILE_W, height=TILE_H)
            label = cam.replace("_", " ").title() if i == 0 else f"t={i}"
            frame = add_label(frame, f"{cam}" if i == 0 else f"step {i * STEPS_BETWEEN_CAPTURES}")
            tiles.append(frame)

        row = np.concatenate(tiles, axis=1)
        rows.append(row)

    # Header row with camera names
    header_tiles = []
    for cam in MONTAGE_CAMERAS:
        header = np.zeros((LABEL_H, TILE_W, 3), dtype=np.uint8)
        header[:] = [40, 40, 40]
        img = Image.fromarray(header)
        draw = ImageDraw.Draw(img)
        name = cam.replace("_", " ")
        draw.text((8, 6), name, fill=(220, 220, 220))
        header_tiles.append(np.array(img))
    header_row = np.concatenate(header_tiles, axis=1)

    montage = np.concatenate([header_row] + rows, axis=0)
    return montage


def run_viewer(env):
    """Launch interactive viewer with a scripted policy."""
    print("\nLaunching interactive viewer (close window to exit)...")
    print("  Arm sweeps through joint space — watch the cable swing!")

    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    step = 0
    while viewer.is_running():
        t = step * 0.05
        action = np.zeros(7, dtype=np.float32)

        # Sinusoidal sweeps across joints for visual appeal
        action[0] = 0.4 * np.sin(t * 0.8)          # shoulder pan
        action[1] = 0.3 * np.sin(t * 0.5 + 1.0)    # shoulder lift
        action[2] = 0.35 * np.sin(t * 0.7 + 2.0)   # elbow
        action[3] = 0.5 * np.sin(t * 1.1)           # wrist 1
        action[4] = 0.4 * np.sin(t * 0.9 + 0.5)    # wrist 2
        action[5] = 0.3 * np.sin(t * 1.3 + 1.5)    # wrist 3
        action[6] = np.sin(t * 0.4)                 # gripper open/close

        env.step(action)
        viewer.sync()
        step += 1
        time.sleep(0.02)

    viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true",
                        help="Open interactive 3D viewer after saving montage")
    args = parser.parse_args()

    print("Building UR5e cable insertion env...")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=False)
    env.reset()
    print(f"  Model: {env.model.nbody} bodies, {env.model.njnt} joints, "
          f"{env.model.nu} actuators")

    # --- Montage ---
    print(f"\nRendering {MONTAGE_STEPS}-step montage "
          f"({len(MONTAGE_CAMERAS)} cameras, {STEPS_BETWEEN_CAPTURES} "
          f"sub-steps between captures)...")
    montage = make_montage(env)
    FRAMES_DIR.mkdir(exist_ok=True)
    out_path = FRAMES_DIR / "ur5e_montage.png"
    Image.fromarray(montage).save(out_path)
    print(f"  Saved: {out_path}  ({montage.shape[1]}x{montage.shape[0]})")

    # --- Interactive viewer ---
    if args.viewer:
        env.reset()
        run_viewer(env)
    else:
        print("\nTip: run with --viewer for interactive 3D exploration")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
