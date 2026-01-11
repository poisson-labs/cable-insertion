from stable_baselines3 import PPO
from cable_env import CableInsertionEnv

# STEP 1: Verify baseline still works (no randomization)
# Once this works, we'll add randomization with same hyperparams

env = CableInsertionEnv(randomize=True)  # Now with domain randomization

print("Training WITH domain randomization")
print("Using same hyperparameters that worked on fixed task")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    ent_coef=0.01,        # Original value that worked
    learning_rate=0.0001,  # Original value that worked
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

model.learn(total_timesteps=4_000_000)  # Longer training for better convergence

model.save("cable_ppo")
print("\nTraining done! Saved: cable_ppo.zip")
