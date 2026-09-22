"""Load UR5e + Robotiq 2F-85 from MuJoCo Menagerie and verify they work.

Renders offscreen frames of:
  1. UR5e arm only (home pose)
  2. UR5e + Robotiq 2F-85 composed via MjSpec.attach()
  3. Joint-space trajectory + gripper open/close to confirm actuation

Usage:
    python experiments/test_ur5e.py           # headless, saves PNGs
    python experiments/test_ur5e.py --viewer   # opens interactive viewer
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
from PIL import Image

THIRD_PARTY = ROOT / "third_party"
UR5E_SCENE = str(THIRD_PARTY / "universal_robots_ur5e" / "scene.xml")
UR5E_XML = str(THIRD_PARTY / "universal_robots_ur5e" / "ur5e.xml")
GRIPPER_XML = str(THIRD_PARTY / "robotiq_2f85" / "2f85.xml")

FRAMES_DIR = ROOT / "experiments" / "frames"


def render_frame(model, data, width=640, height=480):
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data)
    frame = renderer.render().copy()
    renderer.close()
    return frame


def save(frame, name):
    FRAMES_DIR.mkdir(exist_ok=True)
    path = FRAMES_DIR / name
    Image.fromarray(frame).save(path)
    print(f"  Saved: {path}")


def test_ur5e_only():
    """Test 1: Load UR5e arm standalone at home pose."""
    print("=== Test 1: UR5e arm only ===")
    model = mujoco.MjModel.from_xml_path(UR5E_SCENE)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(
        f"  Joints: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}"
    )
    print(f"  Home qpos: {data.qpos.round(3)}")

    save(render_frame(model, data), "ur5e_home.png")
    return model, data


def compose_ur5e_gripper():
    """Use MjSpec.attach() to mount the Robotiq 2F-85 on the UR5e's attachment_site."""
    arm = mujoco.MjSpec.from_file(UR5E_XML)
    gripper = mujoco.MjSpec.from_file(GRIPPER_XML)

    # The UR5e defines an attachment_site on wrist_3_link
    site = arm.site("attachment_site")

    # Attach with prefix to avoid name collisions
    arm.attach(gripper, "gripper-", "", site=site)

    # Add scene elements (floor + light)
    floor = arm.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.2, 0.3, 0.4, 1]

    light = arm.worldbody.add_light()
    light.pos = [0, 0, 1.5]
    light.dir = [0, 0, -1]

    return arm.compile()


def test_composed():
    """Test 2: Load composed UR5e + Robotiq 2F-85."""
    print("\n=== Test 2: UR5e + Robotiq 2F-85 (MjSpec.attach) ===")

    model = compose_ur5e_gripper()
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print(f"  Actuators: {act_names}")

    save(render_frame(model, data), "ur5e_gripper.png")
    return model, data


def test_actuation(model, data):
    """Test 3: Move arm joints + open/close gripper."""
    print("\n=== Test 3: Joint actuation + gripper ===")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    home_ctrl = data.ctrl.copy()

    # Find gripper actuator index
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper-fingers_actuator")

    frames = []
    n_steps = 100
    for i in range(n_steps):
        t = i / n_steps * 2 * np.pi
        data.ctrl[:] = home_ctrl

        # Arm: sweep shoulder_pan and elbow
        data.ctrl[0] = home_ctrl[0] + 0.5 * np.sin(t)  # shoulder_pan
        data.ctrl[2] = home_ctrl[2] + 0.3 * np.sin(2 * t)  # elbow

        # Gripper: open/close cycle (0=open, 255=closed)
        data.ctrl[grip_id] = 127.5 + 127.5 * np.sin(t)

        for _ in range(20):
            mujoco.mj_step(model, data)

        if i % 25 == 0:
            arm_q = " ".join(f"{q:.2f}" for q in data.qpos[:6])
            grip_val = data.ctrl[grip_id]
            print(f"  Step {i:3d}: arm=[{arm_q}]  gripper={grip_val:.0f}/255")
            frames.append(render_frame(model, data))

    for idx, frame in enumerate(frames):
        save(frame, f"ur5e_motion_{idx}.png")

    print(f"  Saved {len(frames)} motion frames")


def run_viewer(model, data):
    """Open interactive viewer."""
    import mujoco.viewer

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    print("\n=== Interactive viewer (close window to exit) ===")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Open interactive MuJoCo viewer")
    args = parser.parse_args()

    test_ur5e_only()
    model, data = test_composed()
    test_actuation(model, data)

    if args.viewer:
        run_viewer(model, data)

    print("\nAll tests passed.")
