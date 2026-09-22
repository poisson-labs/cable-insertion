import numpy as np
import mujoco

from envs.cable_env import SCENE_XML


def test_direct_actuator_control():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    ctrl_target = np.array([0.1, 0.0, -0.05], dtype=np.float32)
    data.ctrl[0] = ctrl_target[0]
    data.ctrl[1] = ctrl_target[1]
    data.ctrl[2] = ctrl_target[2]

    # Headless simulation stepping
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Verify slide joints tracked actuator control
    np.testing.assert_allclose(data.qpos[0:3], ctrl_target, atol=0.01)

    # sensordata contains connector [0:3] and gripper [3:6]
    assert len(data.sensordata) >= 6
    conn = data.sensordata[0:3]
    grip = data.sensordata[3:6]

    # Verify sensor coordinates are finite
    assert np.all(np.isfinite(conn))
    assert np.all(np.isfinite(grip))
