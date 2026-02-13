import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.cable_env import CableInsertionEnv

env = CableInsertionEnv(obs_mode="vision", randomize=True)

print("Vision-based training (3 cameras + state)")
print(f"Observation space: {env.observation_space}")

models_dir = ROOT / "models"
logs_dir = ROOT / "logs"
models_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path=str(models_dir),
    name_prefix="cable_vision_ppo",
)

model = PPO(
    "MultiInputPolicy",       # auto CNN for images, MLP for state
    env,
    verbose=1,
    ent_coef=0.01,
    learning_rate=3e-4,        # SB3 default, good starting point for CNN
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log=str(logs_dir),
)

model.learn(
    total_timesteps=4_000_000,
    callback=checkpoint_cb,
)

model.save(str(models_dir / "cable_vision_ppo"))
print(f"\nTraining done! Saved: {models_dir / 'cable_vision_ppo.zip'}")
