import numpy as np
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("cable_scene.xml")
data = mujoco.MjData(model)

target = np.array([0.25, 0, 0.02])

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        data.ctrl[0] = target[0]
        data.ctrl[1] = target[1]
        data.ctrl[2] = target[2]
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if step % 500 == 0:
            connector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "connector")
            connector_pos = data.xpos[connector_id]
            print(f"connector pos: {connector_pos[0]:.3f}, {connector_pos[1]:.3f}, {connector_pos[2]:.3f}")
            print(f"socket is at:  0.150, 0.000, 0.050")
            print()
        step += 1
        time.sleep(0.002)

