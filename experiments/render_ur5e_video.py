"""Render a 30-second demo video of the UR5e cable insertion env.

Main view: wide third-person camera at 720p
PiP panels: 3 wrist cameras in the bottom-right corner

Usage:
    python experiments/render_ur5e_video.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import imageio
from envs.cable_env_ur5e import CableInsertionUR5eEnv

# Video settings
FPS = 20
DURATION = 30  # seconds
N_FRAMES = FPS * DURATION  # 600

# Main camera
MAIN_W, MAIN_H = 1280, 720
MAIN_CAM = "third_person"

# PiP cameras
PIP_CAMS = ["wrist_left", "wrist_center", "wrist_right"]
PIP_W, PIP_H = 200, 150
PIP_BORDER = 2  # border thickness around each pip
PIP_GAP = 8  # gap between pip panels
PIP_MARGIN = 16  # margin from frame edge

OUT_PATH = ROOT / "experiments" / "frames" / "ur5e_demo_v6.mp4"


def scripted_action(t):
    """Smooth sinusoidal joint sweeps + gripper cycling."""
    a = np.zeros(7, dtype=np.float32)

    # Phase 1 (0-10s): gentle exploration, gripper open
    # Phase 2 (10-20s): bigger sweeps, gripper cycling
    # Phase 3 (20-30s): approach-like motion, gripper closes

    if t < 10:
        a[0] = 0.35 * np.sin(t * 0.8)
        a[1] = 0.25 * np.sin(t * 0.5 + 1.0)
        a[2] = 0.30 * np.sin(t * 0.7 + 2.0)
        a[3] = 0.20 * np.sin(t * 1.1)
        a[4] = 0.15 * np.sin(t * 0.9)
        a[5] = 0.10 * np.sin(t * 1.3)
        a[6] = -0.8  # mostly open
    elif t < 20:
        a[0] = 0.50 * np.sin(t * 0.6 + 0.5)
        a[1] = 0.40 * np.sin(t * 0.4 + 1.5)
        a[2] = 0.45 * np.sin(t * 0.9 + 0.3)
        a[3] = 0.60 * np.sin(t * 1.2 + 1.0)
        a[4] = 0.50 * np.sin(t * 0.7 + 2.0)
        a[5] = 0.40 * np.sin(t * 1.5 + 0.5)
        a[6] = np.sin(t * 0.6)  # cycle open/close
    else:
        # Slowly converge toward less motion + closed gripper
        decay = 1.0 - (t - 20) / 15.0
        decay = max(decay, 0.2)
        a[0] = decay * 0.30 * np.sin(t * 0.5)
        a[1] = decay * 0.35 * np.sin(t * 0.3 + 1.0)
        a[2] = decay * 0.25 * np.sin(t * 0.6 + 2.0)
        a[3] = decay * 0.40 * np.sin(t * 0.8)
        a[4] = decay * 0.30 * np.sin(t * 1.0 + 0.5)
        a[5] = decay * 0.20 * np.sin(t * 0.7 + 1.5)
        a[6] = 0.6 + 0.4 * np.sin(t * 0.4)  # mostly closed

    return a


def composite_pip(main_frame, pip_frames):
    """Overlay PiP panels onto the top-right of the main frame."""
    frame = main_frame.copy()
    n = len(pip_frames)

    # Total PiP strip width
    strip_w = n * (PIP_W + 2 * PIP_BORDER) + (n - 1) * PIP_GAP
    strip_h = PIP_H + 2 * PIP_BORDER

    # Position: top-right
    x_start = MAIN_W - PIP_MARGIN - strip_w
    y_start = PIP_MARGIN

    for i, pip in enumerate(pip_frames):
        x = x_start + i * (PIP_W + 2 * PIP_BORDER + PIP_GAP)
        y = y_start

        # Draw border (dark background)
        frame[y : y + strip_h, x : x + PIP_W + 2 * PIP_BORDER] = 20

        # Paste pip image inside border
        frame[y + PIP_BORDER : y + PIP_BORDER + PIP_H, x + PIP_BORDER : x + PIP_BORDER + PIP_W] = (
            pip
        )

    return frame


def main():
    print("Building UR5e env...")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=False)
    env.reset()
    print(f"  Model: {env.model.nbody} bodies, {env.model.njnt} joints")
    print(f"  Video: {MAIN_W}x{MAIN_H} @ {FPS}fps, {DURATION}s ({N_FRAMES} frames)")
    print(f"  PiP:   {len(PIP_CAMS)} cameras @ {PIP_W}x{PIP_H}")

    OUT_PATH.parent.mkdir(exist_ok=True)

    t0 = time.time()
    print(f"\nRendering to {OUT_PATH} ...")

    writer = imageio.get_writer(
        str(OUT_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,  # high quality
        pixelformat="yuv420p",  # broad compatibility
    )

    for i in range(N_FRAMES):
        sim_t = i / FPS
        action = scripted_action(sim_t)
        env.step(action)

        # Main camera
        main = env.render_camera(MAIN_CAM, width=MAIN_W, height=MAIN_H)

        # PiP cameras
        pips = [env.render_camera(c, width=PIP_W, height=PIP_H) for c in PIP_CAMS]

        # Composite
        frame = composite_pip(main, pips)
        writer.append_data(frame)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps_render = (i + 1) / elapsed
            eta = (N_FRAMES - i - 1) / fps_render
            print(f"  frame {i + 1}/{N_FRAMES}  ({fps_render:.1f} render-fps, ETA {eta:.0f}s)")

    writer.close()

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total ({N_FRAMES / elapsed:.1f} render-fps)")
    print(f"Saved: {OUT_PATH}")
    env.close()


if __name__ == "__main__":
    main()
