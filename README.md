# 🎮 Formative 2 Assignment: Deep Q-Learning - Training and Playing an RL Agent

In this assignment, you will use **Stable Baselines3** and **Gymnasium** to train and evaluate a **Deep Q-Network (DQN)** agent. The agent will learn to play an **Atari game**, and you'll evaluate its performance afterward.

---

## 🕹️ Environment Selection

Choose any **Atari environment** from the Gymnasium collection:  
🔗 [Atari Environments (Gymnasium)](https://gymnasium.farama.org/environments/atari/)

---

## 📁 File Overview

### `train.py`
- Trains a DQN agent
- Compares `MLPPolicy` vs. `CnnPolicy`
- Saves the trained model to `dqn_model.zip`
- Logs reward trends and episode length
- Explores different hyperparameters

### `play.py`
- Loads `dqn_model.zip`
- Runs the agent in the same Atari environment
- Uses `GreedyQPolicy` for evaluation
- Displays game in real-time with `env.render()`

---

## 🧠 Hyperparameter Tuning

You should test multiple sets of hyperparameters and **record the results** in the table below.

### 🔧 Hyperparameter Configuration Table

| **Hyperparameter Set** | **Noted Behavior** |
|------------------------|--------------------|
| `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
| `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
| `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
| `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |
| `lr=, gamma=, batch= , epsilon_start=, epsilon_end=, epsilon_decay=` | |

> ℹ️ You should fill in the table with results such as:  
> - Smoother learning  
> - Faster convergence  
> - Poor exploration  
> - Overfitting, etc.

---

## 🚀 How to Run

### ✅ Install dependencies

```bash
pip install gymnasium[atari]
pip install stable-baselines3[extra]
