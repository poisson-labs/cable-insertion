import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class CableInsertionEnv(gym.Env):
    def __init__(self, max_steps=200):
        self.model = mujoco.MjModel.from_xml_path("cable_scene.xml")
        self.data = mujoco.MjData(self.model)

        # Socket position (target)
        self.target = np.array([0.18, 0, 0.04])

        # Gripper body initial position (from XML)
        # Actions are world coords, ctrl needs joint offsets
        self.gripper_origin = np.array([0.1, 0.0, 0.2])

        # Episode management
        self.max_steps = max_steps
        self.current_step = 0
        self.prev_dist = None

        # Get body IDs for velocity lookup
        self.connector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "connector")

        # Action: gripper x, y, z target positions (world coordinates)
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.3, 0.0], dtype=np.float32),
            high=np.array([0.4, 0.3, 0.6], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: connector pos (3) + connector vel (3) + gripper pos (3) + target (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        self.prev_dist = None
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # Convert world coords to joint offsets
        ctrl = np.array(action) - self.gripper_origin
        self.data.ctrl[:] = ctrl

        # Step simulation (100 steps = 100ms of sim time with 0.001s timestep)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        connector_pos = self.data.sensordata[0:3]
        dist = np.linalg.norm(connector_pos - self.target)

        # Base reward: negative distance
        reward = -dist * 10

        # Shaping: reward for getting closer
        if self.prev_dist is not None:
            improvement = self.prev_dist - dist
            reward += improvement * 50  # Bonus for reducing distance
        self.prev_dist = dist

        # Success
        done = dist <= 0.02
        if done:
            reward += 100.0

        # Timeout (truncation, not termination)
        truncated = self.current_step >= self.max_steps

        return obs, reward, done, truncated, {"distance": dist}

    def _get_obs(self):
        connector_pos = self.data.sensordata[0:3]
        gripper_pos = self.data.sensordata[3:6]
        connector_vel = self.data.cvel[self.connector_id][3:6]  # linear velocity

        return np.concatenate([
            connector_pos,
            connector_vel,
            gripper_pos,
            self.target
        ]).astype(np.float32)
