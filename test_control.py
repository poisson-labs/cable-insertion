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
            # sensordata is flat: [connector x,y,z, gripper x,y,z]
            conn = data.sensordata[0:3]
            grip = data.sensordata[3:6]
            dist = np.linalg.norm(conn - np.array([0.15, 0, 0.05]))
            print(f"connector: {conn.round(3)}, gripper: {grip.round(3)}, dist to socket: {dist:.3f}m")
        
        step += 1
        time.sleep(0.002)