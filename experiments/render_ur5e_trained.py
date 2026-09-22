"""Render trained UR5e policy running 5 episodes over 30 seconds.

Main view: wide third-person camera at 720p
PiP panels: 3 wrist cameras in the top-right corner

Usage:
    python experiments/render_ur5e_trained.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import imageio
from stable_baselines3 import PPO
from envs.cable_env_ur5e import CableInsertionUR5eEnv

# Video settings
FPS = 20
DURATION = 30  # seconds
N_FRAMES = FPS * DURATION  # 600
N_EPISODES = 5

# Main camera
MAIN_W, MAIN_H = 1280, 720
MAIN_CAM = "third_person"

# PiP cameras
PIP_CAMS = ["wrist_left", "wrist_center", "wrist_right"]
PIP_W, PIP_H = 200, 150
PIP_BORDER = 2
PIP_GAP = 8
PIP_MARGIN = 16

MODEL_PATH = ROOT / "models" / "ur5e_state_2M.zip"
OUT_PATH = ROOT / "experiments" / "frames" / "ur5e_trained.mp4"

# Pause frames after success before resetting
PAUSE_AFTER_SUCCESS = int(FPS * 1.0)  # 1 second hold


def composite_pip(main_frame, pip_frames):
    """Overlay PiP panels onto the top-right of the main frame."""
    frame = main_frame.copy()
    n = len(pip_frames)

    strip_w = n * (PIP_W + 2 * PIP_BORDER) + (n - 1) * PIP_GAP
    strip_h = PIP_H + 2 * PIP_BORDER

    x_start = MAIN_W - PIP_MARGIN - strip_w
    y_start = PIP_MARGIN

    for i, pip in enumerate(pip_frames):
        x = x_start + i * (PIP_W + 2 * PIP_BORDER + PIP_GAP)
        y = y_start

        frame[y : y + strip_h, x : x + PIP_W + 2 * PIP_BORDER] = 20
        frame[y + PIP_BORDER : y + PIP_BORDER + PIP_H, x + PIP_BORDER : x + PIP_BORDER + PIP_W] = (
            pip
        )

    return frame


def burn_hud(frame, episode, step, dist, done):
    """Burn episode/step/distance HUD into the top-left corner."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Semi-transparent bar
    draw.rectangle([0, 0, 340, 32], fill=(0, 0, 0, 200))

    status = "SUCCESS" if done else f"{dist * 100:.1f}cm"
    color = (100, 255, 100) if done else (255, 255, 255)
    text = f"Episode {episode}/{N_EPISODES}  Step {step:3d}  Dist: {status}"
    draw.text((10, 8), text, fill=color)

    return np.array(img)


def main():
    print("Building UR5e env...")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=True)
    print(f"  Model: {env.model.nbody} bodies, {env.model.njnt} joints")

    print(f"Loading policy: {MODEL_PATH.name}")
    model = PPO.load(str(MODEL_PATH), env=env)

    print(f"  Video: {MAIN_W}x{MAIN_H} @ {FPS}fps, {DURATION}s ({N_FRAMES} frames)")
    print(f"  PiP:   {len(PIP_CAMS)} cameras @ {PIP_W}x{PIP_H}")
    print(f"  Episodes: {N_EPISODES}")

    OUT_PATH.parent.mkdir(exist_ok=True)

    t0 = time.time()
    print(f"\nRendering to {OUT_PATH} ...")

    writer = imageio.get_writer(
        str(OUT_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    obs, _ = env.reset()
    episode = 1
    ep_step = 0
    ep_done = False
    pause_remaining = 0
    last_dist = 0.0
    ep_stats = []

    for i in range(N_FRAMES):
        if pause_remaining > 0:
            # Hold on the success frame
            pause_remaining -= 1
            if pause_remaining == 0 and episode < N_EPISODES:
                episode += 1
                obs, _ = env.reset()
                ep_step = 0
                ep_done = False
        elif ep_done or ep_step >= 200:
            # Episode ended — start pause or next episode
            ep_stats.append(
                {"ep": episode, "steps": ep_step, "dist": last_dist, "success": ep_done}
            )
            print(
                f"  Episode {episode}: {ep_step} steps, "
                f"dist={last_dist * 100:.2f}cm"
                f"{' SUCCESS' if ep_done else ''}"
            )
            if episode < N_EPISODES:
                pause_remaining = PAUSE_AFTER_SUCCESS
            else:
                # Last episode done — hold remaining frames
                pause_remaining = N_FRAMES  # just hold
        else:
            # Normal policy step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            last_dist = info["distance"]
            ep_step += 1
            ep_done = done

        # Render all cameras
        main = env.render_camera(MAIN_CAM, width=MAIN_W, height=MAIN_H)
        pips = [env.render_camera(c, width=PIP_W, height=PIP_H) for c in PIP_CAMS]

        frame = composite_pip(main, pips)
        frame = burn_hud(frame, episode, ep_step, last_dist, ep_done)
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

    if ep_stats:
        successes = sum(1 for s in ep_stats if s["success"])
        print(f"\nEpisode summary: {successes}/{len(ep_stats)} successes")
        for s in ep_stats:
            tag = "OK" if s["success"] else "FAIL"
            print(f"  ep{s['ep']}: {s['steps']:3d} steps, {s['dist'] * 100:.2f}cm [{tag}]")

    env.close()


if __name__ == "__main__":
    main()
