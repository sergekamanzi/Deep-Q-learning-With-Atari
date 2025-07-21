import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
import time
import os

#defining parameters
ENV_ID = "ALE/Breakout-v5"
MODEL_NAME = "models/dqn_models/dqn_model"
NUM_EPISODES = 15
MAX_STEPS_PER_EPISODE = 1000 #this is to prevent it from looping indefintely

#Adding preprocessing just as during training
def make_env():
    env = gym.make(ENV_ID, render_mode="human")
    env = AtariWrapper(env)
    env = Monitor(env)
    return env
    
#Loading the trained model
def main():
    if not os.path.exists(f"{MODEL_NAME}.zip"):
        raise FileNotFoundError(f"Model file not found at: {MODEL_NAME}.zip")

    env = make_env()
    model = DQN.load(MODEL_NAME, exclude=["replay_buffer"])

    for episode in range(NUM_EPISODES):
        print(f"\n--- Episode {episode + 1} ---")
        obs, _ = env.reset(seed=episode)
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

            time.sleep(0.01)  

        print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f} - Steps: {steps}")

    env.close()

if __name__ == "__main__":
    main()
