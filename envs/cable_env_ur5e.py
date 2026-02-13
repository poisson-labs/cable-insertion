"""Competition-spec cable insertion environment with UR5e + Robotiq 2F-85.

Model is built dynamically via MjSpec composition (no static XML) because
MuJoCo's attach() API requires Python-side model merging.

Action space: 7-dim continuous
  [0:6] - Joint velocity commands (scaled by MAX_JOINT_VEL)
  [6]   - Gripper command (-1=open, +1=closed)

Observation space (vision mode): Dict
  "wrist_left"    - (84, 84, 3) uint8 RGB
  "wrist_center"  - (84, 84, 3) uint8 RGB
  "wrist_right"   - (84, 84, 3) uint8 RGB
  "state"         - (22,) float32:
      joint_pos (6), joint_vel (6), ft_force (3), ft_torque (3),
      gripper_state (1), target_pos (3)

Observation space (state mode): (22,) float32 (same as "state" above)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UR5E_XML = str(ROOT / "third_party" / "universal_robots_ur5e" / "ur5e.xml")
GRIPPER_XML = str(ROOT / "third_party" / "robotiq_2f85" / "2f85.xml")

# Home joint configuration for UR5e
HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

# Vision observation setup
VISION_CAMERAS = ("wrist_left", "wrist_center", "wrist_right")
VISION_SIZE = 84

# Joint velocity limits (rad/s) — conservative for safety
MAX_JOINT_VEL = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], dtype=np.float32)

# Cable parameters
N_CABLE_SEGMENTS = 5
CABLE_SEG_LENGTH = 0.025  # 2.5cm per segment
CABLE_RADIUS = 0.006
CABLE_DAMPING = 0.15
CABLE_STIFFNESS = 0.08

# Simulation timing
PHYSICS_DT = 0.002          # 500Hz physics
N_SUBSTEPS = 25              # 25 substeps = 0.05s control period (20Hz)
CONTROL_DT = PHYSICS_DT * N_SUBSTEPS

DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 480
STATE_DIM = 22  # joint_pos(6) + joint_vel(6) + ft(6) + grip(1) + target(3)


class CableInsertionUR5eEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, max_steps=200, randomize=True, render_mode=None,
                 camera_name="overhead", render_width=DEFAULT_RENDER_WIDTH,
                 render_height=DEFAULT_RENDER_HEIGHT, obs_mode="state"):
        self.max_steps = max_steps
        self.randomize = randomize
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.render_width = render_width
        self.render_height = render_height
        self.obs_mode = obs_mode

        # Build model via MjSpec composition
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)

        # Renderers (created lazily)
        self._renderer = None
        self._obs_renderer = None
        self._viewer = None

        # Cache named element IDs
        self._cache_ids()

        # Initialize home pose and compute default target
        self._reset_to_home()
        self.target_base = self._compute_default_target()
        self.target = self.target_base.copy()

        # Episode state
        self.current_step = 0
        self.thresholds_crossed = set()
        self.default_damping = self.model.dof_damping.copy()

        # Action: 6 joint velocities [-1,1] + 1 gripper [-1,1]
        self.action_space = spaces.Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
        )

        # State: 22-dim
        self._state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        if obs_mode == "state":
            self.observation_space = self._state_space
        elif obs_mode == "vision":
            cam_space = spaces.Box(
                0, 255, shape=(VISION_SIZE, VISION_SIZE, 3), dtype=np.uint8
            )
            self.observation_space = spaces.Dict({
                **{cam: cam_space for cam in VISION_CAMERAS},
                "state": self._state_space,
            })
        else:
            raise ValueError(f"Unknown obs_mode={obs_mode!r}")

    # ------------------------------------------------------------------
    # Model building via MjSpec
    # ------------------------------------------------------------------

    def _build_model(self):
        """Compose UR5e + Robotiq + cable + workspace + sensors."""
        arm = mujoco.MjSpec.from_file(UR5E_XML)
        gripper = mujoco.MjSpec.from_file(GRIPPER_XML)
        site = arm.site("attachment_site")
        arm.attach(gripper, "gripper-", "", site=site)

        # Physics options
        arm.option.timestep = PHYSICS_DT

        self._add_workspace(arm)
        self._add_cameras(arm)
        self._add_cable(arm)
        self._add_sensors(arm)

        return arm.compile()

    def _add_workspace(self, spec):
        """Floor, lights, and visual socket target."""
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [2, 2, 0.01]
        floor.rgba = [0.25, 0.25, 0.25, 1]

        l1 = spec.worldbody.add_light()
        l1.pos = [0.3, 0, 2]
        l1.dir = [0, 0, -1]

        l2 = spec.worldbody.add_light()
        l2.pos = [0.5, 0.5, 1.5]
        l2.dir = [-0.3, -0.3, -1]

        # Socket marker (visual only — no collision)
        socket = spec.worldbody.add_body()
        socket.name = "socket"
        socket.pos = [-0.1, 0.45, 0.02]  # near where cable hangs in home pose
        sg = socket.add_geom()
        sg.name = "socket_geom"
        sg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        sg.size = [0.02, 0.015, 0]
        sg.rgba = [0.2, 0.6, 0.2, 1]
        sg.contype = 0
        sg.conaffinity = 0

    def _add_cameras(self, spec):
        """Overhead, side, and 3 wrist cameras."""
        # Overhead (high up, centered over workspace)
        oc = spec.worldbody.add_camera()
        oc.name = "overhead"
        oc.pos = [-0.1, 0.3, 1.8]
        oc.quat = [1, 0, 0, 0]   # default: looks along -Z (downward)
        oc.fovy = 70

        # Side (front-right, looking toward workspace)
        # Axis (1,1,0)/sqrt(2), angle 90° → looks along (-0.707, 0.707, 0)
        sc = spec.worldbody.add_camera()
        sc.name = "side"
        sc.pos = [0.5, -0.1, 0.35]
        sc.quat = [0.707, 0.5, 0.5, 0]
        sc.fovy = 70

        # Wrist cameras on wrist_3_link
        # Tool axis is +Y in wrist_3_link frame.
        # Camera default looks along -Z. Rotate 90° around X to look along +Y.
        # quat [w,x,y,z] = [cos(45°), sin(45°), 0, 0]
        wrist = spec.body("wrist_3_link")
        tool_quat = [0.7071, 0.7071, 0, 0]

        for name, x_off in [("wrist_center", 0.0),
                            ("wrist_left", -0.04),
                            ("wrist_right", 0.04)]:
            cam = wrist.add_camera()
            cam.name = name
            cam.pos = [x_off, 0.13, 0]  # past attachment site, along tool axis
            cam.quat = tool_quat
            cam.fovy = 60

    def _add_cable(self, spec):
        """Flexible ball-joint cable chain attached to gripper fingertip."""
        gripper_base = spec.body("gripper-base")
        prev = gripper_base

        for i in range(N_CABLE_SEGMENTS):
            seg = prev.add_body()
            seg.name = f"cable_{i}"
            # First segment: offset to just past the pinch site (z=0.145)
            seg.pos = [0, 0, 0.155] if i == 0 else [0, 0, CABLE_SEG_LENGTH]

            j = seg.add_joint()
            j.type = mujoco.mjtJoint.mjJNT_BALL
            j.damping = CABLE_DAMPING
            j.stiffness = CABLE_STIFFNESS

            g = seg.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size = [CABLE_RADIUS, 0, 0]
            g.fromto = [0, 0, 0, 0, 0, CABLE_SEG_LENGTH]
            g.rgba = [0.9, 0.4, 0.1, 1]

            prev = seg

        # Connector tip
        conn = prev.add_body()
        conn.name = "connector"
        conn.pos = [0, 0, CABLE_SEG_LENGTH]
        cj = conn.add_joint()
        cj.type = mujoco.mjtJoint.mjJNT_BALL
        cj.damping = CABLE_DAMPING
        cj.stiffness = CABLE_STIFFNESS
        cg = conn.add_geom()
        cg.name = "connector_geom"
        cg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        cg.size = [0.012, 0.018, 0]
        cg.rgba = [0.3, 0.3, 0.8, 1]

    def _add_sensors(self, spec):
        """Connector position + wrist force/torque sensors."""
        # F/T sensor site on wrist_3_link
        wrist = spec.body("wrist_3_link")
        ft = wrist.add_site()
        ft.name = "ft_sensor_site"
        ft.pos = [0, 0.09, 0]

        # Connector position (sensordata[0:3])
        s1 = spec.add_sensor()
        s1.name = "connector_pos"
        s1.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
        s1.objname = "connector"
        s1.objtype = mujoco.mjtObj.mjOBJ_BODY

        # Wrist force (sensordata[3:6])
        s2 = spec.add_sensor()
        s2.name = "wrist_force"
        s2.type = mujoco.mjtSensor.mjSENS_FORCE
        s2.objname = "ft_sensor_site"
        s2.objtype = mujoco.mjtObj.mjOBJ_SITE

        # Wrist torque (sensordata[6:9])
        s3 = spec.add_sensor()
        s3.name = "wrist_torque"
        s3.type = mujoco.mjtSensor.mjSENS_TORQUE
        s3.objname = "ft_sensor_site"
        s3.objtype = mujoco.mjtObj.mjOBJ_SITE

    # ------------------------------------------------------------------
    # ID caching
    # ------------------------------------------------------------------

    def _cache_ids(self):
        m = self.model
        self.connector_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "connector")
        self.socket_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "socket")
        self.grip_act_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper-fingers_actuator")

        # Sensor data slices (order matches _add_sensors)
        self.sens_conn_pos = slice(0, 3)
        self.sens_force = slice(3, 6)
        self.sens_torque = slice(6, 9)

        # Arm joint indices in qpos/qvel (first 6 hinge joints)
        self.arm_qpos = slice(0, 6)
        self.arm_qvel = slice(0, 6)

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def _reset_to_home(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_QPOS
        self.data.ctrl[:6] = HOME_QPOS
        self.data.ctrl[self.grip_act_id] = 0  # open
        mujoco.mj_forward(self.model, self.data)

    def _compute_default_target(self):
        """Place default target near the resting connector position."""
        conn = self.data.sensordata[self.sens_conn_pos].copy()
        conn[2] = 0.02  # near floor
        return conn

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._reset_to_home()

        if self.randomize:
            self._apply_domain_randomization()

        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        self.thresholds_crossed = set()
        return self._get_obs(), {}

    def _apply_domain_randomization(self):
        rng = self.np_random

        # Target: ±3cm XY, ±1cm Z
        noise = rng.uniform([-0.03, -0.03, -0.01], [0.03, 0.03, 0.01])
        self.target = self.target_base + noise
        self.model.body_pos[self.socket_id] = self.target

        # Arm pose: ±5° per joint
        joint_noise = rng.uniform(-0.087, 0.087, size=6)
        self.data.qpos[:6] = HOME_QPOS + joint_noise
        self.data.ctrl[:6] = self.data.qpos[:6]

        # Cable damping: ±20%
        scale = rng.uniform(0.8, 1.2)
        self.model.dof_damping[:] = self.default_damping * scale

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)

        # Joint velocity control: ctrl_target = current_pos + vel * dt
        joint_vel = action[:6] * MAX_JOINT_VEL
        self.data.ctrl[:6] = self.data.qpos[self.arm_qpos] + joint_vel * CONTROL_DT

        # Gripper: [-1,1] → [0,255]
        grip_cmd = np.clip(action[6], -1.0, 1.0)
        self.data.ctrl[self.grip_act_id] = (float(grip_cmd) + 1.0) / 2.0 * 255.0

        # Step physics
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Reward: distance-based with graduated bonuses
        connector_pos = self.data.sensordata[self.sens_conn_pos]
        dist = float(np.linalg.norm(connector_pos - self.target))

        reward = -dist * 10.0

        if dist <= 0.05 and 5 not in self.thresholds_crossed:
            reward += 10.0
            self.thresholds_crossed.add(5)
        if dist <= 0.03 and 3 not in self.thresholds_crossed:
            reward += 25.0
            self.thresholds_crossed.add(3)
        if dist <= 0.02 and 2 not in self.thresholds_crossed:
            reward += 100.0
            self.thresholds_crossed.add(2)

        done = bool(dist <= 0.02)
        truncated = bool(self.current_step >= self.max_steps)

        return obs, reward, done, truncated, {"distance": dist}

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self):
        state = self._get_state_obs()
        if self.obs_mode == "state":
            return state
        return self._get_vision_obs(state)

    def _get_state_obs(self):
        d = self.data
        joint_pos = d.qpos[self.arm_qpos].copy()       # 6
        joint_vel = d.qvel[self.arm_qvel].copy()        # 6
        ft_force = d.sensordata[self.sens_force].copy()  # 3
        ft_torque = d.sensordata[self.sens_torque].copy() # 3
        grip = np.array([d.ctrl[self.grip_act_id] / 255.0])  # 1
        return np.concatenate([
            joint_pos, joint_vel, ft_force, ft_torque, grip, self.target
        ]).astype(np.float32)

    def _get_vision_obs(self, state):
        if self._obs_renderer is None:
            self._obs_renderer = mujoco.Renderer(
                self.model, height=VISION_SIZE, width=VISION_SIZE
            )
        obs = {}
        for cam in VISION_CAMERAS:
            self._obs_renderer.update_scene(self.data, camera=cam)
            obs[cam] = self._obs_renderer.render().copy()
        obs["state"] = state
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_camera(self.camera_name)
        elif self.render_mode == "human":
            self._render_human()
            return None

    def render_camera(self, camera_name, width=None, height=None):
        w = width or self.render_width
        h = height or self.render_height

        if self._renderer is not None:
            rw, rh = self._renderer._width, self._renderer._height
            if rw != w or rh != h:
                self._renderer.close()
                self._renderer = None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)

        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def _render_human(self):
        if self._viewer is None or not self._viewer.is_running():
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._obs_renderer is not None:
            self._obs_renderer.close()
            self._obs_renderer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
