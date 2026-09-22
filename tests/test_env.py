import numpy as np

from envs.cable_env import CableInsertionEnv


def test_cable_env_initialization():
    env = CableInsertionEnv()
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (12,)
    assert env.max_steps == 200


def test_cable_env_reset():
    env = CableInsertionEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (12,)
    assert isinstance(info, dict)


def test_cable_env_step():
    env = CableInsertionEnv()
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info
    assert info["distance"] > 0


def test_cable_env_deterministic_reset_with_seed():
    env1 = CableInsertionEnv(randomize=True)
    env2 = CableInsertionEnv(randomize=True)
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2, atol=1e-5)
