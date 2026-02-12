import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

SCENE_XML = str(Path(__file__).parent / "assets" / "cable_scene.xml")

class CableInsertionEnv(gym.Env):
    def __init__(self, max_steps=200, randomize=True):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        # Base positions (from XML)
        self.target_base = np.array([0.18, 0.0, 0.04])
        self.gripper_origin = np.array([0.1, 0.0, 0.2])

        # Current target (will be randomized each reset)
        self.target = self.target_base.copy()

        # Episode management
        self.max_steps = max_steps
        self.current_step = 0

        # Track which distance thresholds we've crossed (for graduated bonuses)
        self.thresholds_crossed = set()

        # Domain randomization flag
        self.randomize = randomize

        # Get body/joint IDs for randomization and observation
        self.connector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "connector")
        self.socket_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "socket")

        # Find ball joint indices for physics randomization
        # Ball joints are indices 3+ (after x, y, z slide joints)
        self.ball_joint_ids = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_BALL:
                self.ball_joint_ids.append(i)

        # Store default physics params (to randomize around)
        # dof_damping is per-DOF, ball joints have 3 DOFs each
        self.default_damping = self.model.dof_damping.copy()

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

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)

        if self.randomize:
            self._apply_domain_randomization()

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.thresholds_crossed = set()
        return self._get_obs(), {}

    def _apply_domain_randomization(self):
        """
        Randomize initial conditions and physics each episode.
        This forces the policy to generalize rather than memorize.
        """
        rng = self.np_random  # Gymnasium's seeded RNG

        # 1. RANDOMIZE TARGET POSITION
        #    Conservative: ±1cm in x/y, ±0.5cm in z
        target_noise = rng.uniform(
            low=[-0.01, -0.01, -0.005],
            high=[0.01, 0.01, 0.005]
        )
        self.target = self.target_base + target_noise

        # Also move the visual socket body to match
        self.model.body_pos[self.socket_id] = self.target

        # 2. RANDOMIZE INITIAL GRIPPER POSITION
        #    Conservative: ±2cm in x/z, ±1cm in y
        gripper_noise = rng.uniform(
            low=[-0.02, -0.01, -0.02],
            high=[0.02, 0.01, 0.02]
        )
        self.data.qpos[0:3] = gripper_noise

        # 3. RANDOMIZE CABLE PHYSICS
        #    Conservative: ±15% variation
        damping_scale = rng.uniform(0.85, 1.15)
        self.model.dof_damping[:] = self.default_damping * damping_scale

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

        # Base reward: negative distance only
        reward = -dist * 10

        # No improvement shaping - let graduated bonuses guide learning
        # (Removed: improvement * 50 bonus was conflicting with threshold bonuses)

        # GRADUATED SUCCESS BONUSES
        # Give one-time bonuses when crossing thresholds (only once per episode)
        # This creates intermediate goals on the way to success
        if dist <= 0.05 and 5 not in self.thresholds_crossed:
            reward += 10.0   # Small bonus for getting within 5cm
            self.thresholds_crossed.add(5)
        if dist <= 0.03 and 3 not in self.thresholds_crossed:
            reward += 25.0   # Medium bonus for getting within 3cm
            self.thresholds_crossed.add(3)
        if dist <= 0.02 and 2 not in self.thresholds_crossed:
            reward += 100.0  # Big bonus for success (within 2cm)
            self.thresholds_crossed.add(2)

        # Episode ends on success
        done = dist <= 0.02

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
            self.target  # Now includes the randomized target!
        ]).astype(np.float32)
