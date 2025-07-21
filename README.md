# Formative 2 Assignment: Deep Q-Learning - Training and Playing an RL Agent

In here, we  used **Stable Baselines3** and **Gymnasium** to train and evaluate a **Deep Q-Network (DQN)** agent and  the agent will learn to play an **Atari game**

## Environment Selection

**Atari environment** chosen: BREAKOUT
🔗 [Atari Environments (Gymnasium)](https://ale.farama.org/environments/breakout/)


### Hyperparameter Configuration Table

| **Hyperparameter Set** | **Noted Behavior** |
|------------------------|--------------------|
|'**jesse**' `lr=1e-4, gamma=0.99, batch=32 , epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.1, total_timesteps=200_000` |The model was beginning to learn strategies to succeed but learning was unstable because the total reward varied highly. |
|'**serge**' `lr=1e-4, gamma=0.99, batch=64 , epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1, total_timesteps=100_000` |so the timesteps i used here were little and usually for break-out game we use 1M but cause of the time to train was much bigger and the also the capacity was to large to handle, for buffersize using 10,000 was very low and also its good to start with 100k but also it requires much RAM. |
|'**adediwura**' `lr=2e-4, gamma=0.99, batch=32 , epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1, total_timesteps=150_000` |There were a few episodes with zero rewards. |
|'**bernice**' `lr= 1e-3, gamma=0.95, batch= 32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay= 0.1, total_timesteps=300000` |With epsilon_start=1.0 and a  decay of 0.1, the agent explores heavily early on, leading to inconsistent rewards between 1 and 6 across Episodes 1–15. The moderate gamma=0.95 encourages long-term reward, but learning is still unstable due to the exploitation value used.A lower  epsilon decay or more training steps could have improve performance by giving the agent more time to explore effectively before settling into exploitation. I initially started training on 1 million timesteps but Kaggle kept on crashing, compute difficulties, so I had to reduce the timesteps to be able train successfully|

### Comparison of CNN with MLP for Atari Breakout
A CNN policy works better than an MLP policy for Breakout because:

1. CNNs are designed for image inputs: Breakout observations are pixel-based frames (2D images). CNNs can extract spatial patterns like edges, shapes, and motion across frames.

2. Spatial locality: CNNs capture local visual features (e.g., ball location, paddle, bricks) and hierarchically build higher-level representations important for decision-making.

3. MLPs ignore spatial structure: An MLP flattens the image, losing spatial relationships between pixels. This makes it harder to learn useful visual features.

## How to Run

### Install dependencies

```
pip install gymnasium[atari]
pip install stable-baselines3[extra]
```

### Run scripts
To train:
```
python train.py
```

To play:
```
python play.py
```
