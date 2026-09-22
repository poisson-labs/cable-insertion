import numpy as np

from envs.cable_env import CableInsertionEnv


def test_reachability_settles_and_tracks():
    env = CableInsertionEnv(randomize=False)
    env.reset(seed=42)

    # Command gripper target towards the socket location
    target_action = np.array([0.20, 0.0, 0.05], dtype=np.float32)
    for _ in range(50):
        obs, reward, done, truncated, info = env.step(target_action)

    final_dist = info["distance"]
    # The connector settles within reach of the target socket
    assert np.isfinite(final_dist)
    assert final_dist < 0.25
