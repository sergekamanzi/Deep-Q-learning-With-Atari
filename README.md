# Formative 2 Assignment: Deep Q-Learning - Training and Playing an RL Agent

In here, we  used **Stable Baselines3** and **Gymnasium** to train and evaluate a **Deep Q-Network (DQN)** agent and  the agent will learn to play an **Atari game**

## Environment Selection

**Atari environment** we choose: BOWLING 
🔗 [Atari Environments (Gymnasium)](https://gymnasium.farama.org/environments/atari/)


### Hyperparameter Configuration Table

| **Hyperparameter Set** | **Noted Behavior** |
|------------------------|--------------------|
|'jesse' `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
|'serge' `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
|'adediwura' `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
|'bernice' `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |


## How to Run

### Install dependencies

```bash
pip install gymnasium[atari]
pip install stable-baselines3[extra]
