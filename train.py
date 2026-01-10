from stable_baselines3 import PPO
from cable_env import CableInsertionEnv

env = CableInsertionEnv()

obs, _ = env.reset()
action = [0.25, 0, 0.02]  # What we found earlier
for i in range(100):
    obs, reward, done, _, info = env.step(action)
    print(f"dist: {info['distance']:.4f}")
    if done:
        print("SUCCESS!")
        break

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    ent_coef=0.01,
    learning_rate=0.0001,
)
model.learn(total_timesteps=2000000)

model.save("cable_ppo")
print("Training done, saved to cable_ppo.zip")