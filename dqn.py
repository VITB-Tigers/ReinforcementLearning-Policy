# METADATA [dqn.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: This code snippet implements a Deep Q-Network (DQN) for the Snake Game using TensorFlow and Keras. 
    # The Snake is trained to learn optimal movement strategies based on rewards and penalties using the DQN algorithm. 
    # The objective is to avoid collisions while eating randomly placed apples. The model uses a neural network to approximate 
    # the Q-value function and improve the agent's performance over time.

    # Developed By: 
        # Name: Mohini Tiwari
        # Role: Developer
        # Code ownership rights: Mohini Tiwari

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Dependencies:
        # Python 3.11.5
        # Libraries:
            # Pygame 2.4.0
            # Numpy 1.24.4
            # Matplotlib 3.8.0
            # TensorFlow 2.14.0
            # Keras 2.14.0

# Importing the necessary libraries
import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Constants for game settings
WIDTH, HEIGHT = 640, 480      # Width and height of the game window
BLOCK_SIZE = 20                # Size of each block in the grid
WHITE = (255, 255, 255)        # Color for the game over text
GREEN = (0, 255, 0)            # Color for the snake
RED = (255, 0, 0)              # Color for the food
BLACK = (0, 0, 0)              # Color for the background

# Snake game class
class SnakeGame:
    def __init__(self):
        pygame.init()  # Initialize pygame
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))  # Create game window
        pygame.display.set_caption('Snake AI')  # Set window title
        self.clock = pygame.time.Clock()  # Create a clock to control the game's frame rate
        self.reset()  # Reset the game to its initial state

    def reset(self):
        """Reset the game state."""
        self.snake = [(WIDTH // 2, HEIGHT // 2)]  # Initialize the snake in the middle of the window
        self.direction = (BLOCK_SIZE, 0)  # Set initial direction (moving right)
        self.spawn_food()  # Spawn the first piece of food
        self.score = 0  # Initialize score

    def spawn_food(self):
        """Spawn food in a random position that is not occupied by the snake."""
        self.food = (random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                     random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
        # Ensure food does not spawn on the snake
        while self.food in self.snake:
            self.food = (random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                         random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)

    def step(self, action):
        """Execute a game step based on the action taken by the agent."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Allow the game to exit
                pygame.quit()

        self.move_snake(action)  # Move the snake based on the chosen action

        # Insert new head of the snake
        self.snake.insert(0, self.new_head)

        # Check if the snake has eaten the food
        if self.snake[0] == self.food:  
            self.score += 10  # Increase score for eating food
            self.spawn_food()  # Spawn new food
            reward = 10  # Positive reward for eating
        else:
            self.snake.pop()  # Remove the tail if food was not eaten
            reward = -1  # Small negative reward for moving without eating

        # Check for collisions
        done = self.is_collision()
        if done:
            reward = -10  # Penalty for colliding with walls or itself

        # Update the user interface
        self.update_ui()
        self.clock.tick(10)  # Control the frame rate

        return reward, done, self.score  # Return the reward, game over status, and current score

    def move_snake(self, action):
        """Move the snake based on the action taken."""
        # Actions: 0 = forward, 1 = right, 2 = left
        if action == 1:  # Turn right
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:  # Turn left
            self.direction = (self.direction[1], -self.direction[0])

        # Calculate the new head position
        x, y = self.snake[0]
        self.new_head = (x + self.direction[0], y + self.direction[1])

    def is_collision(self):
        """Check for collisions with walls or the snake itself."""
        x, y = self.snake[0]
        # Check for wall collision and self-collision
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or (x, y) in self.snake[1:]:
            return True
        return False

    def update_ui(self):
        """Update the game display."""
        self.display.fill(BLACK)  # Fill background with black
        for x, y in self.snake:  # Draw the snake
            pygame.draw.rect(self.display, GREEN, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))  # Draw food
        pygame.display.flip()  # Update the display

    def get_state(self):
        """Get the current state of the game as a flat array."""
        head_x, head_y = self.snake[0]  # Get snake's head position
        food_x, food_y = self.food  # Get food position
        
        # Normalize direction to grid coordinates
        direction = np.array(self.direction) // BLOCK_SIZE
        state = np.array([
            head_x // BLOCK_SIZE,  # Head X position
            head_y // BLOCK_SIZE,  # Head Y position
            food_x // BLOCK_SIZE,  # Food X position
            food_y // BLOCK_SIZE,  # Food Y position
            direction[0],  # Direction X component
            direction[1]   # Direction Y component
        ])
        return state  # Return the state array

# DQN Agent Class
class DQNAgent:
    def __init__(self):
        self.state_size = 6  # Number of state features
        self.action_size = 3  # Number of possible actions
        self.memory = deque(maxlen=5000)  # Memory for experience replay
        self.gamma = 0.95  # Discount rate for future rewards
        self.epsilon = 1.0  # Exploration rate (initially high for exploration)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay factor for exploration rate
        self.learning_rate = 0.001  # Learning rate for the neural network
        self.model = self._build_model()  # Build the neural network model

    def _build_model(self):
        """Build the neural network model for Q-value prediction."""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))  # Input layer
        model.add(Dense(64, activation='relu'))  # Hidden layer
        model.add(Dense(32, activation='relu'))  # Hidden layer
        model.add(Dense(self.action_size, activation='linear'))  # Output layer (Q-values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Compile the model
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Decide on an action to take based on the current state."""
        if np.random.rand() <= self.epsilon:  # Check for exploration
            return random.randrange(self.action_size)  # Random action
        act_values = self.model.predict(state)  # Predict Q-values for the given state
        return np.argmax(act_values[0])  # Return the action with the highest Q-value

    def replay(self, batch_size):
        """Train the model using a random sample of experiences from memory."""
        minibatch = random.sample(self.memory, batch_size)  # Sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward  # Initialize target with immediate reward
            if not done:  # If not done, calculate target using the Q-value formula
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)  # Get the current Q-values for the state
            target_f[0][action] = target  # Update the Q-value for the taken action
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the model on the updated target
        if self.epsilon > self.epsilon_min:  # Decay exploration rate
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)

    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)

# Function to show the game over screen
def show_game_over(game):
    font = pygame.font.SysFont(None, 55)  # Create font object for game over text
    game_over_surface = font.render('Game Over!', True, WHITE)  # Render game over text
    game.display.blit(game_over_surface, (WIDTH // 2 - 100, HEIGHT // 2))  # Draw text on the display
    pygame.display.flip()  # Update the display
    pygame.time.wait(2000)  # Wait for 2 seconds before closing

# Main loop for the game with episode control
def main(episode_number=None):
    game = SnakeGame()  # Create a new game instance
    agent = DQNAgent()  # Create a new DQN agent
    episodes = episode_number if episode_number is not None else 10  # Set number of episodes
    scores = []  # List to store scores for each episode
    max_scores = []  # List to store maximum scores
    losses = []  # List to store losses during training

    try:
        for e in range(episodes):
            game.reset()  # Reset the game at the start of each episode
            state = game.get_state().reshape(1, agent.state_size)  # Get the initial state
            score = 0  # Initialize score
            print(f"Starting episode {e + 1}/{episodes}")

            while True:
                action = agent.act(state)  # Decide on an action
                reward, done, current_score = game.step(action)  # Execute the action and get feedback

                next_state = game.get_state().reshape(1, agent.state_size)  # Get the next state
                agent.remember(state, action, reward, next_state, done)  # Store experience in memory

                state = next_state  # Move to the next state
                score += reward  # Accumulate score using the reward

                if done:  # If the game is over
                    scores.append(score)  # Store the score of this episode
                    max_scores.append(max(scores) if scores else score)  # Store the maximum score
                    print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon}")
                    show_game_over(game)  # Show game over screen
                    break

            # Train the agent if enough experiences are available
            if len(agent.memory) > 64:
                agent.replay(64)  # Increased batch size for better learning

            # Save model weights periodically
            if e % 100 == 0:
                agent.save(f"snake_dqn_{e}.weights.h5")

        plot_performance(scores, max_scores)  # Plot performance after training

    except KeyboardInterrupt:
        print("Training interrupted")  # Handle interruption gracefully

def plot_performance(scores, max_scores):
    """Function to plot the performance metrics."""
    plt.figure(figsize=(12, 6))  # Set figure size for the plot
    plt.plot(scores, label='Score per Episode', color='blue')  # Plot scores
    plt.plot(max_scores, label='Max Score', color='green')  # Plot maximum scores
    plt.xlabel('Episodes')  # X-axis label
    plt.ylabel('Scores')  # Y-axis label
    plt.title('DQN Agent Performance in Snake Game')  # Plot title
    plt.legend()  # Show legend
    plt.show()  # Display the plot

if __name__ == "__main__":
    main(episode_number=1000)  # Run the main function with 1000 episodes
