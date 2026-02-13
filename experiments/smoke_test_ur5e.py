"""Smoke test for CableInsertionUR5eEnv.

Tests:
  1. Model builds and compiles (MjSpec composition)
  2. reset() returns correct observation shape
  3. step() with random actions works
  4. Sensor data flows (connector pos, F/T)
  5. Vision mode observations
  6. Render produces frames
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from envs.cable_env_ur5e import CableInsertionUR5eEnv, STATE_DIM, VISION_CAMERAS, VISION_SIZE


def test_state_mode():
    print("=== Test 1: State mode — build, reset, step ===")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=False)

    print(f"  Model: nbody={env.model.nbody}, njnt={env.model.njnt}, "
          f"nu={env.model.nu}, nsensor={env.model.nsensor}")
    print(f"  nq={env.model.nq}, nv={env.model.nv}")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs space: {env.observation_space}")

    obs, info = env.reset()
    print(f"\n  reset() obs shape: {obs.shape} (expected ({STATE_DIM},))")
    assert obs.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {obs.shape}"

    print(f"  obs breakdown:")
    print(f"    joint_pos:  {obs[0:6].round(3)}")
    print(f"    joint_vel:  {obs[6:12].round(3)}")
    print(f"    ft_force:   {obs[12:15].round(3)}")
    print(f"    ft_torque:  {obs[15:18].round(3)}")
    print(f"    grip_state: {obs[18]:.3f}")
    print(f"    target_pos: {obs[19:22].round(3)}")

    # Step with zero action (hold position)
    obs2, reward, done, truncated, info = env.step(np.zeros(7))
    print(f"\n  step(zeros) → reward={reward:.3f}, dist={info['distance']:.4f}")
    assert obs2.shape == (STATE_DIM,)

    # Step with random actions
    print(f"\n  Running 20 random steps...")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if i % 5 == 0:
            print(f"    step {i+1}: dist={info['distance']:.4f}, reward={reward:.2f}")
    print("  OK")
    env.close()
    return True


def test_vision_mode():
    print("\n=== Test 2: Vision mode ===")
    env = CableInsertionUR5eEnv(obs_mode="vision", randomize=False)

    obs, info = env.reset()
    assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"

    for cam in VISION_CAMERAS:
        assert cam in obs, f"Missing camera '{cam}' in obs"
        img = obs[cam]
        assert img.shape == (VISION_SIZE, VISION_SIZE, 3), f"Bad shape for {cam}: {img.shape}"
        assert img.dtype == np.uint8
        pct_black = np.mean(img.sum(axis=-1) == 0) * 100
        print(f"  {cam}: shape={img.shape}, dtype={img.dtype}, black={pct_black:.0f}%")

    assert "state" in obs
    assert obs["state"].shape == (STATE_DIM,)
    print(f"  state: shape={obs['state'].shape}")

    # One step
    obs2, reward, done, truncated, info = env.step(np.zeros(7))
    assert isinstance(obs2, dict)
    print("  step() OK")

    env.close()
    return True


def test_render():
    print("\n=== Test 3: Render frames ===")
    env = CableInsertionUR5eEnv(
        obs_mode="state", render_mode="rgb_array", randomize=False
    )
    env.reset()

    cameras = ["overhead", "side", "wrist_center", "wrist_left", "wrist_right"]
    frames_dir = ROOT / "experiments" / "frames"
    frames_dir.mkdir(exist_ok=True)

    for cam_name in cameras:
        frame = env.render_camera(cam_name)
        print(f"  {cam_name}: shape={frame.shape}, "
              f"black={np.mean(frame.sum(axis=-1) == 0) * 100:.0f}%")

        # Save frame
        from PIL import Image
        Image.fromarray(frame).save(frames_dir / f"ur5e_{cam_name}.png")

    print(f"  Saved to {frames_dir}/ur5e_*.png")
    env.close()
    return True


def test_domain_randomization():
    print("\n=== Test 4: Domain randomization ===")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=True)

    targets = []
    for i in range(5):
        obs, _ = env.reset(seed=i)
        targets.append(obs[19:22].copy())
        print(f"  seed={i}: target={targets[-1].round(4)}")

    # Targets should differ across seeds
    diffs = [np.linalg.norm(targets[i] - targets[0]) for i in range(1, 5)]
    assert any(d > 0.001 for d in diffs), "Targets not varying!"
    print("  Targets vary across resets ✓")
    env.close()
    return True


def test_episode():
    print("\n=== Test 5: Full episode (200 steps) ===")
    env = CableInsertionUR5eEnv(obs_mode="state", randomize=False, max_steps=200)
    obs, _ = env.reset()

    total_reward = 0
    min_dist = float("inf")
    for step in range(200):
        action = env.action_space.sample() * 0.1  # small random actions
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        min_dist = min(min_dist, info["distance"])
        if done or truncated:
            break

    print(f"  Finished at step {step+1}, total_reward={total_reward:.1f}, "
          f"min_dist={min_dist:.4f}")
    print(f"  done={done}, truncated={truncated}")
    env.close()
    return True


if __name__ == "__main__":
    results = []
    for test_fn in [test_state_mode, test_vision_mode, test_render,
                    test_domain_randomization, test_episode]:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n=== Summary ===")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")

    if all(ok for _, ok in results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
