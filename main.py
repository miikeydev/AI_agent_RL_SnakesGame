import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Configuration variables for the game

snake_speed = 2000
window_x = 800
window_y = 800
pixel_size = 40
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Initialize pygame

pygame.init()
pygame.display.set_caption('Snakes with DQL')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()


# Helper functions for state representation and checking dangers
def get_relative_direction(snake_position, fruit_position):

    # Get the relative direction of the fruit from the snake's head.

    if snake_position[0] < fruit_position[0]:  # fruit is to the right
        fruit_dir_x = 1
    elif snake_position[0] > fruit_position[0]:  # fruit is to the left
        fruit_dir_x = -1
    else:
        fruit_dir_x = 0

    if snake_position[1] < fruit_position[1]:  # fruit is below
        fruit_dir_y = 1
    elif snake_position[1] > fruit_position[1]:  # fruit is above
        fruit_dir_y = -1
    else:
        fruit_dir_y = 0

    return fruit_dir_x, fruit_dir_y


def check_dangers(snake_position, snake_body, direction, window_x, window_y):

    # Check for dangers in the immediate front, left, and right based on the current direction.

    front_danger, left_danger, right_danger = 0, 0, 0
    current_x, current_y = snake_position

    # Check danger in front

    if direction == 'UP':
        front_cell = (current_x, current_y - pixel_size)
    elif direction == 'DOWN':
        front_cell = (current_x, current_y + pixel_size)
    elif direction == 'LEFT':
        front_cell = (current_x - pixel_size, current_y)
    else:  # 'RIGHT'
        front_cell = (current_x + pixel_size, current_y)

    if (front_cell in snake_body) or (
            front_cell[0] < 0 or front_cell[0] >= window_x or front_cell[1] < 0 or front_cell[1] >= window_y):
        front_danger = 1

    # Check danger to the left (this will depend on the current direction)

    if direction == 'UP':
        left_cell = (current_x - pixel_size, current_y)
    elif direction == 'DOWN':
        left_cell = (current_x + pixel_size, current_y)
    elif direction == 'LEFT':
        left_cell = (current_x, current_y + pixel_size)
    else:  # 'RIGHT'
        left_cell = (current_x, current_y - pixel_size)

    if (left_cell in snake_body) or (
            left_cell[0] < 0 or left_cell[0] >= window_x or left_cell[1] < 0 or left_cell[1] >= window_y):
        left_danger = 1

    # Check danger to the right (this will also depend on the current direction)

    if direction == 'UP':
        right_cell = (current_x + pixel_size, current_y)
    elif direction == 'DOWN':
        right_cell = (current_x - pixel_size, current_y)
    elif direction == 'LEFT':
        right_cell = (current_x, current_y - pixel_size)
    else:  # 'RIGHT'
        right_cell = (current_x, current_y + pixel_size)

    if (right_cell in snake_body) or (
            right_cell[0] < 0 or right_cell[0] >= window_x or right_cell[1] < 0 or right_cell[1] >= window_y):
        right_danger = 1

    return front_danger, left_danger, right_danger


def get_state(snake_position, snake_body, fruit_position, direction, window_x, window_y):

    # Convert the game state into the defined representation.

    # Initialize state as a list of zeros
    state = [0] * 11

    # Set the direction of the snake
    if direction == 'UP':
        state[0] = 1
    elif direction == 'DOWN':
        state[1] = 1
    elif direction == 'LEFT':
        state[2] = 1
    elif direction == 'RIGHT':
        state[3] = 1

    # Set the relative direction of the fruit
    fruit_dir_x, fruit_dir_y = get_relative_direction(snake_position, fruit_position)
    if fruit_dir_x == -1:
        state[4] = 1  # Fruit is to the left
    elif fruit_dir_x == 1:
        state[5] = 1  # Fruit is to the right

    if fruit_dir_y == -1:
        state[6] = 1  # Fruit is above
    elif fruit_dir_y == 1:
        state[7] = 1  # Fruit is below

    # Check for dangers
    front_danger, left_danger, right_danger = check_dangers(snake_position, snake_body, direction, window_x, window_y)
    state[8], state[9], state[10] = front_danger, left_danger, right_danger

    return state



class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, output_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQL Agent
class DQAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995):

        self.net = DQNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayMemory(1000000)
        self.batch_size = 100

    def select_action(self, state):
        # Select an action using epsilon-greedy policy.
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])  # Random action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.net(state_tensor)
                return torch.argmax(q_values).item()  # Best action

    def learn(self):
        # Learn from a batch of experiences.

        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)

        # Extract states, actions, rewards, and next states from the experiences
        states, actions, rewards, next_states = zip(*experiences)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Compute Q-values
        current_q_values = self.net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values)
        target_q_values = target_q_values.unsqueeze(1)

        # Compute loss and update the network
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# Training loop
def game_training(agent, num_episodes=1000):
    episode_rewards = []
    episode_lengths = []
    best_score = 0

    for episode in range(num_episodes):

        # Only display the game every episode, you can change the value if you want to see less or more game :)
        display_game = (episode % 1 == 0)

        # Game initialization
        snake_start = [window_x // 2, window_y // 2]
        snake_position = snake_start.copy()
        snake_body = [snake_start, [snake_start[0] - pixel_size, snake_start[1]],[snake_start[0] - 2 * pixel_size, snake_start[1]]]
        fruit_position = [random.randrange(1, (window_x // pixel_size)) * pixel_size,random.randrange(1, (window_y // pixel_size)) * pixel_size]

        score = 0
        direction = 'RIGHT'
        change_to = direction

        cumulative_reward = 0

        exit_game = False
        multiplier = 1

        # Start game loop
        while True:
            reward = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            state = get_state(snake_position, snake_body, fruit_position, direction, window_x, window_y)

            # Agent selects action

            action = agent.select_action(state)
            if action == 0:  # Go straight
                pass
            elif action == 1:  # Turn left
                if direction == 'UP':
                    change_to = 'LEFT'
                if direction == 'DOWN':
                    change_to = 'RIGHT'
                if direction == 'LEFT':
                    change_to = 'DOWN'
                if direction == 'RIGHT':
                    change_to = 'UP'
            elif action == 2:  # Turn right
                if direction == 'UP':
                    change_to = 'RIGHT'
                if direction == 'DOWN':
                    change_to = 'LEFT'
                if direction == 'LEFT':
                    change_to = 'UP'
                if direction == 'RIGHT':
                    change_to = 'DOWN'

            # Change direction of snake

            if change_to == 'UP':
                direction = 'UP'
            if change_to == 'DOWN':
                direction = 'DOWN'
            if change_to == 'LEFT':
                direction = 'LEFT'
            if change_to == 'RIGHT':
                direction = 'RIGHT'

            # Move the snake

            if direction == 'UP':
                snake_position[1] -= pixel_size
            if direction == 'DOWN':
                snake_position[1] += pixel_size
            if direction == 'LEFT':
                snake_position[0] -= pixel_size
            if direction == 'RIGHT':
                snake_position[0] += pixel_size


            snake_head_rect = pygame.Rect(snake_position[0], snake_position[1], pixel_size, pixel_size)
            fruit_rect = pygame.Rect(fruit_position[0], fruit_position[1], pixel_size, pixel_size)

            snake_body.insert(0, list(snake_position))
            if snake_head_rect.colliderect(fruit_rect):
                score += 10
                reward += 15
                fruit_position = [random.randrange(1, (window_x // pixel_size)) * pixel_size,
                                  random.randrange(1, (window_y // pixel_size)) * pixel_size]
                multiplier += 1
            else:
                snake_body.pop()

            if display_game:
                game_window.fill(black)
                for pos in snake_body:
                    pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], pixel_size, pixel_size))
                pygame.draw.rect(game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], pixel_size, pixel_size))
                pygame.display.flip()



            # Game Over conditions

            if snake_position[0] < 0 or snake_position[0] > window_x - pixel_size:
                reward -= 10
                next_state = get_state(snake_position, snake_body, fruit_position, direction, window_x, window_y)
                agent.memory.push(state, action, reward, next_state)  # Store experience immediately upon game over
                agent.learn()
                cumulative_reward += reward
                print("Game over as snake touched wall", reward)
                break

            if snake_position[1] < 0 or snake_position[1] > window_y - pixel_size:
                reward -= 10
                next_state = get_state(snake_position, snake_body, fruit_position, direction, window_x, window_y)
                agent.memory.push(state, action, reward, next_state)  # Store experience immediately upon game over
                agent.learn()
                cumulative_reward += reward
                print("Game over as snake touched wall", reward)
                break

            for segment in snake_body[1:]:
                if snake_position[0] == segment[0] and snake_position[1] == segment[1]:
                    reward -= 10 * multiplier
                    next_state = get_state(snake_position, snake_body, fruit_position, direction, window_x, window_y)
                    agent.memory.push(state, action, reward, next_state)  # Store experience immediately upon game over
                    agent.learn()
                    cumulative_reward += reward
                    print("Game over as snake touched his own body", reward)
                    exit_game = True
                    break

            if exit_game:
                break

            next_state = get_state(snake_position, snake_body, fruit_position, direction, window_x, window_y)
            cumulative_reward += reward
            agent.memory.push(state, action, reward, next_state)
            agent.learn()

            fps.tick(snake_speed)
        if score > best_score:
            best_score = score


        episode_rewards.append(score)
        episode_lengths.append(len(snake_body) - 3)
        print(f"Episode {episode + 1}/{num_episodes} - Score: {score} - Length: {len(snake_body) - 3} - Cumulative Reward: {cumulative_reward} The best score achieved is: {best_score}")

    return episode_rewards, episode_lengths


# Initialize the DQL agent
agent = DQAgent(11, 3)

episode_rewards, episode_lengths = game_training(agent, num_episodes=1000)

torch.save(agent.net.state_dict(), 'snake_model_weights.pth')



