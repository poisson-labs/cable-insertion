"""Render smooth trained-policy video with substep rendering and orbit camera.

- Renders every substep (25 per control step) for fluid arm motion
- Single continuous episode, target randomizes on each success
- Slow orbit camera around the workspace
- 60fps for buttery playback, ~8x slow-mo of sim time

Usage:
    python experiments/render_ur5e_trained_smooth.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from envs.cable_env_ur5e import CableInsertionUR5eEnv

# ── Video settings ──────────────────────────────────────
FPS = 60
DURATION = 30
N_FRAMES = FPS * DURATION  # 1800

# Main camera (orbit — ID looked up at runtime)
MAIN_W, MAIN_H = 1280, 720
MAIN_CAM = "third_person"

# PiP cameras
PIP_CAMS = ["wrist_left", "wrist_center", "wrist_right"]
PIP_W, PIP_H = 200, 150
PIP_BORDER = 2
PIP_GAP = 8
PIP_MARGIN = 16

# Physics
N_SUBSTEPS = 25  # must match env
FRAMES_PER_CONTROL = N_SUBSTEPS  # render every substep
MAX_CONTROL_STEPS = N_FRAMES // FRAMES_PER_CONTROL + 1

# Orbit camera
ORBIT_CENTER = np.array([-0.05, 0.40, 0.15])
ORBIT_RADIUS = 0.85
ORBIT_HEIGHT_BASE = 0.65
ORBIT_HEIGHT_AMP = 0.10  # gentle vertical bob
ORBIT_REVOLUTIONS = 0.35  # ~126° arc over 30s
ORBIT_START_ANGLE = -0.6  # radians, starting position

# Success / target
SUCCESS_DIST = 0.02
FREEZE_FRAMES = int(FPS * 1.5)  # 1.5s hold on success before reset

MODEL_PATH = ROOT / "models" / "ur5e_state_2M.zip"
OUT_PATH = ROOT / "experiments" / "frames" / "ur5e_trained_smooth.mp4"


# ── Helpers ─────────────────────────────────────────────


def lookat_quat(cam_pos, target, world_up=np.array([0.0, 0.0, 1.0])):
    """Compute MuJoCo camera quaternion [w,x,y,z] for look-at."""
    forward = np.asarray(target, dtype=float) - np.asarray(cam_pos, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.column_stack([right, up, -forward])
    # Rotation matrix → quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(1.0 + trace)
        w, x, y, z = (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )
    return np.array([w, x, y, z])


def update_orbit_camera(model, cam_id, frame_idx):
    """Smoothly orbit the main camera around the workspace."""
    t = frame_idx / N_FRAMES  # 0 → 1
    angle = ORBIT_START_ANGLE + t * ORBIT_REVOLUTIONS * 2 * np.pi
    height = ORBIT_HEIGHT_BASE + ORBIT_HEIGHT_AMP * np.sin(t * np.pi)

    pos = ORBIT_CENTER + np.array(
        [
            ORBIT_RADIUS * np.cos(angle),
            ORBIT_RADIUS * np.sin(angle),
            height,
        ]
    )
    model.cam_pos[cam_id] = pos
    model.cam_quat[cam_id] = lookat_quat(pos, ORBIT_CENTER)


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


def burn_hud(frame, targets_reached, dist, is_success):
    """Burn a minimal HUD into the top-left corner."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 320, 32], fill=(0, 0, 0, 200))

    if is_success:
        text = f"Targets reached: {targets_reached}  INSERTING"
        color = (100, 255, 100)
    else:
        text = f"Targets reached: {targets_reached}  Dist: {dist * 100:.1f}cm"
        color = (255, 255, 255)
    draw.text((10, 8), text, fill=color)
    return np.array(img)


def randomize_target(env, rng):
    """Move socket to a new random position."""
    noise = rng.uniform([-0.03, -0.03, -0.01], [0.03, 0.03, 0.01])
    env.target = env.target_base + noise
    env.model.body_pos[env.socket_id] = env.target
    mujoco.mj_forward(env.model, env.data)


# ── Env internals we replicate for manual stepping ──────
from envs.cable_env_ur5e import MAX_JOINT_VEL, CONTROL_DT


def apply_action(env, action):
    """Set ctrl from action (replicates env.step control logic)."""
    action = np.asarray(action, dtype=np.float32)
    joint_vel = action[:6] * MAX_JOINT_VEL
    env.data.ctrl[:6] = env.data.qpos[env.arm_qpos] + joint_vel * CONTROL_DT
    # Override gripper to closed — policy doesn't use it yet but
    # clamped fingers look much better on video than splayed open.
    env.data.ctrl[env.grip_act_id] = 255.0


def get_dist(env):
    """Current connector-to-target distance."""
    conn = env.data.sensordata[env.sens_conn_pos]
    return float(np.linalg.norm(conn - env.target))


# ── Main ────────────────────────────────────────────────


def main():
    print("Building UR5e env...")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=True)
    obs, _ = env.reset()

    print(f"Loading policy: {MODEL_PATH.name}")
    model_ppo = PPO.load(str(MODEL_PATH), env=env)

    m, d = env.model, env.data
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, MAIN_CAM)

    print(f"  Video: {MAIN_W}x{MAIN_H} @ {FPS}fps, {DURATION}s ({N_FRAMES} frames)")
    print(f"  Substep rendering: {FRAMES_PER_CONTROL} frames per control step")
    print(f"  Max control steps: {MAX_CONTROL_STEPS}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    t0 = time.time()
    print(f"\nRendering to {OUT_PATH} ...")

    writer = imageio.get_writer(
        str(OUT_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    frame_idx = 0
    targets_reached = 0
    control_step = 0
    is_success = False
    freeze_remaining = 0
    substep_in_control = 0
    dist = get_dist(env)

    while frame_idx < N_FRAMES:
        if freeze_remaining > 0:
            # ── Frozen: hold the success pose, no physics ──
            freeze_remaining -= 1

            if freeze_remaining == 0:
                # Reset: new target, keep going
                randomize_target(env, rng)
                obs = env._get_obs()
                dist = get_dist(env)
                is_success = False
                substep_in_control = 0
        else:
            # ── Active: run policy + physics ──
            if substep_in_control == 0:
                action, _ = model_ppo.predict(obs, deterministic=True)
                apply_action(env, action)
                control_step += 1

            mujoco.mj_step(m, d)
            substep_in_control += 1

            # End of control step — check distance
            if substep_in_control >= FRAMES_PER_CONTROL:
                substep_in_control = 0
                obs = env._get_obs()
                dist = get_dist(env)

                if dist < SUCCESS_DIST:
                    targets_reached += 1
                    print(
                        f"  Target {targets_reached} reached at control step "
                        f"{control_step} (dist={dist * 100:.2f}cm)"
                    )
                    is_success = True
                    freeze_remaining = FREEZE_FRAMES

        # ── Render frame (always, even during freeze) ──
        update_orbit_camera(m, cam_id, frame_idx)

        main_frame = env.render_camera(MAIN_CAM, width=MAIN_W, height=MAIN_H)
        pips = [env.render_camera(c, width=PIP_W, height=PIP_H) for c in PIP_CAMS]

        frame = composite_pip(main_frame, pips)
        frame = burn_hud(frame, targets_reached, dist, is_success)
        writer.append_data(frame)
        frame_idx += 1

        if frame_idx % 300 == 0:
            elapsed = time.time() - t0
            fps_r = frame_idx / elapsed
            eta = (N_FRAMES - frame_idx) / fps_r
            print(f"  frame {frame_idx}/{N_FRAMES}  ({fps_r:.1f} render-fps, ETA {eta:.0f}s)")

    writer.close()

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total ({N_FRAMES / elapsed:.1f} render-fps)")
    print(f"Saved: {OUT_PATH}")
    print(f"Targets reached: {targets_reached}")
    print(f"Control steps used: {control_step}")
    print(
        f"Sim time: {control_step * CONTROL_DT:.2f}s shown over {DURATION}s video "
        f"({DURATION / (control_step * CONTROL_DT):.1f}x slow-mo)"
    )
    env.close()


if __name__ == "__main__":
    main()
