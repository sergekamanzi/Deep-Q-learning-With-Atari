import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

#defining parameters
ENV_ID = "ALE/Breakout-v5"
MODEL_DIR = "models/dqn_models"
LOG_DIR = "logs/"
TOTAL_TIMESTEPS = 500000

#Creating environment with preprocessing to reduce noise and complexities to help the model learn faster
def make_env():
    env = gym.make(ENV_ID, render_mode=None)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env]) 
env_eval = DummyVecEnv([make_env]) 

# setting checkpoint to evaluate model at every 50000 time steps
checkpoint_callback = CheckpointCallback(
    save_freq= 50000,
    save_path=MODEL_DIR,
    name_prefix="dqn_model"
)

#defining model and setting hyperparameters
model = DQN(
    "CnnPolicy",        
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=10000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log=LOG_DIR
)

#training the model
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

#Saving the final model
model.save(os.path.join(MODEL_DIR, "dqn_model"))
print("Final model saved.")

# Evaluating the overall trained model
mean_reward, std_reward = evaluate_policy(
    model,
    env_eval,
    n_eval_episodes=15,
    deterministic=True
)
print(f"Final reward: {mean_reward:.2f} ± {std_reward:.2f}")
