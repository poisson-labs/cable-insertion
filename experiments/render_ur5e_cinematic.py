"""Cinematic 30s video of the trained vision policy with camera cuts.

- Hard cuts between cameras on each target reach
- Title card intro
- Substep rendering at 60fps for fluid motion
- Slow orbit on third_person shots
- Minimal HUD

Usage:
    python experiments/render_ur5e_cinematic.py
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
from envs.cable_env_ur5e import CableInsertionUR5eEnv, MAX_JOINT_VEL, CONTROL_DT

# ── Video settings ──────────────────────────────────────
FPS = 60
DURATION = 30
N_FRAMES = FPS * DURATION
W, H = 1280, 720

# ── Shot list: cycle through these on each reach ────────
SHOT_SEQUENCE = [
    "third_person",  # wide establishing
    "wrist_center",  # insertion POV
    "side",  # front-right profile
    "overhead",  # top-down
    "wrist_left",  # left fixed
    "third_person",  # back to wide
    "wrist_right",  # right fixed
]

# ── Timing ──────────────────────────────────────────────
TITLE_FRAMES = int(FPS * 2.5)  # 2.5s title card
FREEZE_FRAMES = int(FPS * 1.2)  # 1.2s hold on success
FADE_FRAMES = int(FPS * 0.3)  # 0.3s fade from black on cut
N_SUBSTEPS = 25
SUCCESS_DIST = 0.02

# ── Orbit camera ────────────────────────────────────────
ORBIT_CENTER = np.array([-0.05, 0.40, 0.15])
ORBIT_RADIUS = 0.85
ORBIT_HEIGHT = 0.65
ORBIT_SPEED = 0.3  # radians per 30s of video
ORBIT_START = -0.6

MODEL_PATH = ROOT / "models" / "ur5e_vision_4M.zip"
OUT_PATH = ROOT / "experiments" / "frames" / "ur5e_cinematic.mp4"


# ── Helpers ─────────────────────────────────────────────


def lookat_quat(cam_pos, target, world_up=np.array([0.0, 0.0, 1.0])):
    forward = np.asarray(target, dtype=float) - np.asarray(cam_pos, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.column_stack([right, up, -forward])
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


def update_orbit(model, cam_id, t_norm):
    """Slowly orbit third_person camera. t_norm is 0→1 over the video."""
    angle = ORBIT_START + t_norm * ORBIT_SPEED * 2 * np.pi
    pos = ORBIT_CENTER + np.array(
        [
            ORBIT_RADIUS * np.cos(angle),
            ORBIT_RADIUS * np.sin(angle),
            ORBIT_HEIGHT,
        ]
    )
    model.cam_pos[cam_id] = pos
    model.cam_quat[cam_id] = lookat_quat(pos, ORBIT_CENTER)


def make_title_card(text_lines):
    """Black frame with centered white text."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = H // 2 - len(text_lines) * 20
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, y), line, fill=(255, 255, 255))
        y += 36
    return np.array(img)


def burn_hud(frame, targets, cam_name, is_success):
    """Minimal HUD: target count + camera label."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Bottom-left: camera name
    draw.rectangle([0, H - 28, 180, H], fill=(0, 0, 0, 180))
    label = cam_name.replace("_", " ").upper()
    draw.text((10, H - 22), label, fill=(160, 160, 160))

    # Top-left: target counter
    draw.rectangle([0, 0, 200, 32], fill=(0, 0, 0, 180))
    color = (100, 255, 100) if is_success else (255, 255, 255)
    status = "INSERTED" if is_success else "TRACKING"
    draw.text((10, 8), f"Targets: {targets}  {status}", fill=color)

    return np.array(img)


def apply_fade(frame, fade_progress):
    """Fade from black. fade_progress: 0=black, 1=full brightness."""
    return (frame * fade_progress).astype(np.uint8)


def apply_action(env, action):
    action = np.asarray(action, dtype=np.float32)
    joint_vel = action[:6] * MAX_JOINT_VEL
    env.data.ctrl[:6] = env.data.qpos[env.arm_qpos] + joint_vel * CONTROL_DT
    env.data.ctrl[env.grip_act_id] = 255.0  # gripper clamped closed


def get_dist(env):
    conn = env.data.sensordata[env.sens_conn_pos]
    return float(np.linalg.norm(conn - env.target))


def randomize_target(env, rng):
    noise = rng.uniform([-0.03, -0.03, -0.01], [0.03, 0.03, 0.01])
    env.target = env.target_base + noise
    env.model.body_pos[env.socket_id] = env.target
    mujoco.mj_forward(env.model, env.data)


# ── Main ────────────────────────────────────────────────


def main():
    print("Building UR5e env...")
    env = CableInsertionUR5eEnv(obs_mode="vision", randomize=True)
    obs, _ = env.reset()

    print(f"Loading policy: {MODEL_PATH.name}")
    model_ppo = PPO.load(str(MODEL_PATH), env=env)

    m, d = env.model, env.data
    tp_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")

    print(f"  Video: {W}x{H} @ {FPS}fps, {DURATION}s ({N_FRAMES} frames)")
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

    # ── Title card ──
    title = make_title_card(
        [
            "UR5e Cable Insertion",
            "",
            "Vision Policy  /  4M Steps  /  100% Success Rate",
        ]
    )
    for i in range(TITLE_FRAMES):
        # Fade in over first 1s, hold, fade out over last 0.5s
        if i < FPS * 1.0:
            alpha = i / (FPS * 1.0)
        elif i > TITLE_FRAMES - FPS * 0.5:
            alpha = (TITLE_FRAMES - i) / (FPS * 0.5)
        else:
            alpha = 1.0
        writer.append_data(apply_fade(title, alpha))

    # ── Policy loop ──
    frame_idx = TITLE_FRAMES
    targets_reached = 0
    shot_idx = 0
    current_cam = SHOT_SEQUENCE[0]
    freeze_remaining = 0
    substep_in_control = 0
    frames_since_cut = 0
    is_success = False
    dist = get_dist(env)

    while frame_idx < N_FRAMES:
        if freeze_remaining > 0:
            # ── Frozen on success pose ──
            freeze_remaining -= 1
            if freeze_remaining == 0:
                # Cut to next camera for next reach
                shot_idx = (shot_idx + 1) % len(SHOT_SEQUENCE)
                current_cam = SHOT_SEQUENCE[shot_idx]
                frames_since_cut = 0
                randomize_target(env, rng)
                obs = env._get_obs()
                dist = get_dist(env)
                is_success = False
                substep_in_control = 0
        else:
            # ── Active policy + physics ──
            if substep_in_control == 0:
                action, _ = model_ppo.predict(obs, deterministic=True)
                apply_action(env, action)

            mujoco.mj_step(m, d)
            substep_in_control += 1

            if substep_in_control >= N_SUBSTEPS:
                substep_in_control = 0
                obs = env._get_obs()
                dist = get_dist(env)

                if dist < SUCCESS_DIST:
                    targets_reached += 1
                    print(
                        f"  Target {targets_reached} reached "
                        f"(dist={dist * 100:.2f}cm, cam={current_cam})"
                    )
                    is_success = True
                    freeze_remaining = FREEZE_FRAMES

        # ── Orbit if on third_person ──
        t_norm = frame_idx / N_FRAMES
        update_orbit(m, tp_cam_id, t_norm)

        # ── Render ──
        raw = env.render_camera(current_cam, width=W, height=H)
        frame = burn_hud(raw, targets_reached, current_cam, is_success)

        # Fade from black on camera cuts
        if frames_since_cut < FADE_FRAMES:
            alpha = frames_since_cut / FADE_FRAMES
            frame = apply_fade(frame, alpha)

        writer.append_data(frame)
        frame_idx += 1
        frames_since_cut += 1

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
    print(
        f"Shots used: {[SHOT_SEQUENCE[i % len(SHOT_SEQUENCE)] for i in range(targets_reached + 1)]}"
    )
    env.close()


if __name__ == "__main__":
    main()
