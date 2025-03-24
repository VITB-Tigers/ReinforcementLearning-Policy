# Snake Game with Deep Q-Learning

This project demonstrates a **Snake Game** implemented using **Deep Q-Learning (DQN)**. The game involves training an AI agent to play the Snake game autonomously. The agent learns through reinforcement learning by interacting with the game environment, improving its score over multiple episodes.

> To Learn more about DQN using your favourite LLM, [click here](prompts.md)!

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Visualizing Results](#visualizing-results)
- [How It Works](#how-it-works)

## Features

- **Deep Q-Learning Agent**: 
  - Utilizes a neural network to predict Q-values for state-action pairs.
  - Implements experience replay for efficient training.
  - Employs an epsilon-greedy policy for exploration and exploitation.

- **Game Environment**: 
  - Built using the `pygame` library.
  - Features grid-based movement, food spawning, and collision handling.

- **Training and Performance Visualization**:
  - Tracks performance metrics like scores and maximum scores per episode.
  - Includes a plotting function to visualize the training results.

## Project Structure
```
/ DQN_SnakeGame
    ├── dqn.py                  # Implementation of the DQN algorithm for training the Snake Game
    ├── prompts.md              # Contains business problem statements and conceptual questions for DQN
    ├── requirements.txt        # Lists all required dependencies to run the DQN implementation
    └── README.md               # Documentation providing an overview of the DQN Snake Game project
```
## Installation

1. Clone the repository from GitHub:
   ```
   git clone https://github.com/VITB-Tigers/ReinforcementLearning-Policy
   ```

2. To create and activate a virtual environment (recommended) using Conda, run the following commands:
   ```
   conda create -n env_name python==3.11.0 -y
   conda activate env_name
   ```
   *Note: This project uses Python 3.11.0*

3. Install the dependencies using:
   ```
   pip install -r requirements.txt
   ```

## How to Run

1. Open a terminal or command prompt in the project directory.

2. Run the script:
   ```
   python dqn.py
   ```

3. Adjust the number of training episodes by modifying the `main()` function:
   ```python
   main(episode_number=1000)
   ```

> The trained model's weights are periodically saved as `.h5` files in the current directory.

## Visualizing Results

After training, the script plots the following metrics:
- **Score per Episode**: The agent's score for each episode.
- **Maximum Score**: The highest score achieved up to each episode.

## How It Works

### Game Logic

1. **Game Environment**:
   - A `SnakeGame` class defines the environment, including snake movement, food spawning, collision detection, and UI updates.
   - The state representation includes:
     - Snake's head position.
     - Food position.
     - Direction of the snake.

2. **Rewards**:
   - **+10** for eating food.
   - **-1** for moving without eating.
   - **-10** for collisions.

3. **Actions**:
   - `0`: Move forward.
   - `1`: Turn right.
   - `2`: Turn left.

### DQN Agent

- **Neural Network Architecture**:
  - Input layer with six features (state size).
  - Two hidden layers with 64 neurons each and ReLU activation.
  - Output layer with three neurons (one for each action).

- **Learning Parameters**:
  - Discount factor (`gamma`): 0.95.
  - Exploration-exploitation trade-off controlled by `epsilon`.
  - Replay memory size: 5000 experiences.
  - Learning rate: 0.001.

### Training Process

- The agent interacts with the environment for a specified number of episodes.
- Experiences are stored in replay memory.
- After each episode, the agent trains on a batch of experiences.

### Performance Tracking

- Scores and maximum scores are tracked for each episode.
- A graph is plotted at the end of training to show the agent's performance over time.
