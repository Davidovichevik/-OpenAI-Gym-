# -OpenAI-Gym-
Reinforcement Learning with OpenAI Gym and Stable-Baselines:
import gym
from stable_baselines3 import PPO

# Create CartPole environment
env = gym.make('CartPole-v1')

# Create PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save('ppo_cartpole')

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
