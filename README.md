# Formative 2 Assignment: Deep Q-Learning - Training and Playing an RL Agent

<img width="638" height="542" alt="Screenshot (425)" src="https://github.com/user-attachments/assets/e80dba26-ee11-42db-8bc3-84329d656506" />


In here, we  used **Stable Baselines3** and **Gymnasium** to train and evaluate a **Deep Q-Network (DQN)** agent and  the agent will learn to play an **Atari game**

## Environment Selection

**Atari environment** chosen: BREAKOUT
🔗 [Atari Environments (Gymnasium)](https://ale.farama.org/environments/breakout/)
<img width="437" height="341" alt="Screenshot 2025-07-22 042252" src="https://github.com/user-attachments/assets/b74e4972-2dcc-41d4-b4de-4615546e7468" />

### Hyperparameter Configuration Table

| **Hyperparameter Set** | **Noted Behavior** |
|------------------------|--------------------|
|'**jesse**' `lr=1e-4, gamma=0.99, batch=32 , epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.1, total_timesteps=500_000` |The model was beginning to learn strategies to succeed but learning was unstable because the total reward varied highly. There was still some difficulty with balancing exploration and exploitation. |
|'**serge**' `lr=1e-4, gamma=0.99, batch=64 , epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1, total_timesteps=100_000` |so the timesteps i used here were little and usually for break-out game we use 1M but cause of the time to train was much bigger and the also the capacity was to large to handle, for buffersize using 10,000 was very low and also its good to start with 100k but also it requires much RAM. |
|'**adediwura**' `lr=2e-4, gamma=0.99, batch=32 , epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1, total_timesteps=50_000` |There were a few episodes with zero rewards. |
|'**bernice**' `lr= 1e-3, gamma=0.95, batch= 32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay= 0.1, total_timesteps=300000` |With epsilon_start=1.0 and a  decay of 0.1, the agent explores heavily early on, leading to inconsistent rewards between 1 and 6 across Episodes 1–15. The moderate gamma=0.95 encourages long-term reward, but learning is still unstable due to the exploitation value used.A lower  epsilon decay or more training steps could have improve performance by giving the agent more time to explore effectively before settling into exploitation. I initially started training on 1 million timesteps but Kaggle kept on crashing, compute difficulties, so I had to reduce the timesteps to be able train successfully|

<img width="349" height="510" alt="Screenshot 2025-07-22 044233" src="https://github.com/user-attachments/assets/2c425fa8-c764-43aa-91d5-632a8b444087" />


### Comparison of CNN with MLP for Atari Breakout
We compared MLPPolicy and CnnPolicy on the Breakout Atari environment to evaluate their performance. The CnnPolicy, built mostly for  image-based input, learned effectively over time with varying reward structure and showing consistent learning growth. In contrast, the MLPPolicy performed poorly: both ep_len_mean and ep_rew_mean were same(e.g., 22.3, 22.3) throughout the whole training, indicating the agent received a reward of 1 per step and failed shortly after, without any meaningful improvement. This suggests the model was unable to extract useful features from the pixel input. We also tried to play the game and then it run into countless errors which majorly because of the shape cause this image was flattened but the input here is in frames.From our research, MLPs are not suitable for raw image data. MLPs are better suited for environments with low-dimensional, structured input such as CartPole, MountainCar, or environments with numerical state vectors. Hence, for breakout; Cnnpolicy is the best.

<img width="901" height="540" alt="Screenshot (424)" src="https://github.com/user-attachments/assets/9ec51a1f-f4bd-489b-9a27-6a6fedd8d9c2" />

Find notebooks here : 
- https://colab.research.google.com/drive/16LHr320ubRvcZn5l4hiGW9A7HBAHiSK_?usp=sharing - Cnnpolicy
- https://colab.research.google.com/drive/1m3Dgzp96oULk1GIG21O9lpJlnNL64zoo?usp=sharing - Mlppolicy


## Group Contribution

- **Jesse**: Worked on `play.py`, adjusted hyperparameters for training, and contributed to the `README`.
- **Bernice**: Worked on `train.py`, adjusted hyperparameters for training, and handled the policy comparison.
- **Serge**: Worked on `train.py` and adjusted hyperparameters for training.
- **Adediwura**: Worked on `play.py` and adjusted hyperparameters for training.

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
