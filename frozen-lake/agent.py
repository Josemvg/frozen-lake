import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Agent():
    """
    Defines an agent that can be used to solve the Frozen Lake environment
    """
    def __init__(self, env, holes=None, frozen=None) -> None:
        """
        Initialize the agent

        Parameters
        ----------
        env : gym environment
            The environment to solve
        holes : list, optional
            The list of holes in the environment, by default None
        frozen : list, optional
            The list of frozen tiles in the environment, by default None
        
        Returns
        -------
        None
            The agent is initialized
        """
        # Initialize the environment
        self.env = env
        # Initialize the Q-table
        self.q_table = np.zeros(
            [self.env.observation_space.n,
            self.env.action_space.n]
        )
        # Initialize the list of frames
        self.frames = []
        # Initialize the list of evaluation frames
        self.eval_frames = []
        # Initialize the list of holes, frozen tiles, rewards and wins
        self.holes = holes
        self.frozen = frozen
        self.rewards = None
        self.wins = None

    def q_train(self, alpha, gamma, epsilon, episodes=1000, modify_rewards=False, save_images=False):
        """
        Train the agent using the Q-learning algorithm

        Parameters
        ----------
        alpha : float
            The learning rate
        gamma : float
            The discount factor
        epsilon : float
            The exploration rate
        episodes : int, optional
            The number of episodes to train for, by default 1000
        modify_rewards : bool, optional
            Whether to modify the rewards, by default False
        save_images : bool, optional
            Whether to save the images, by default False

        Returns
        -------
        None
            The agent is trained
        """
        # Initialize the rewards and wins
        self.rewards = np.zeros(episodes)
        self.wins = np.zeros(episodes)

        # For each episode
        for i in tqdm(range(episodes)):
            state = self.env.reset()[0]
            reward = 0
            done = False
            
            # While the episode is not done
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values

                next_state, reward, done, _, _ = self.env.step(action)

                # Modify the rewards
                if modify_rewards:
                    if next_state in self.holes:
                        reward = reward - 0.2
                    
                    elif next_state in self.frozen:
                        reward = reward - 0.01
                
                # Update the Q-table
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value
                self.rewards[i] += reward

                # Update the wins if the agent reached the goal
                if reward == 1:
                    self.wins[i] += 1

                # Update the state
                state = next_state
                
                # Save the images
                img = self.env.render()
                if save_images:
                    self.frames.append(img)
    
    def evaluate_performance(self, episodes=100, save_images=False):
        """
        Evaluate the performance of the agent

        Parameters
        ----------
        episodes : int, optional
            The number of episodes to evaluate for, by default 100
        save_images : bool, optional
            Whether to save the images, by default False
        
        Returns
        -------
        total_epochs : int
            The total number of epochs
        total_penalties : int
            The total number of penalties
        """
        total_epochs, total_penalties = 0, 0

        # For each episode
        for i in tqdm(range(episodes)):
            state = self.env.reset()[0]
            epochs, penalties, reward = 0, 0, 0
            done = False
            
            # While the episode is not done
            while not done:
                # Take the action with the highest Q-value
                action = np.argmax(self.q_table[state])
                next_state, reward, done, truncated, info = self.env.step(action)
                # Update the penalties if the agent fell in a hole
                if next_state in self.holes:
                    penalties += 1

                # Update the state
                state = next_state

                # Save the image
                img = self.env.render()
                if save_images:
                    self.eval_frames.append(img)

                # Update the epochs
                epochs += 1

            # Update the total epochs and penalties
            total_penalties += penalties
            total_epochs += epochs
        
        return total_epochs, total_penalties

    def plot_q_training_curve(self, window=10):
        """
        Plot the training curve

        Parameters
        ----------
        window : int, optional
            The window size, by default 10

        Returns
        -------
        None
            The training curve is plotted
        """
        avg_rewards = np.array([
            np.mean(self.rewards[i-window:i])  
                if i >= window
                else np.mean(self.rewards[:i])
                for i in range(1, len(self.rewards))
        ])
        
        plt.figure(figsize=(12,8))
        plt.plot(avg_rewards, label='Mean Q Rewards')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.show()
        return

    def plot_q_training_wins(self):
        """
        Plot the training wins

        Returns
        -------
        None
            The training wins are plotted
        """
        plt.figure(figsize=(12, 5))
        plt.xlabel("Run number")
        plt.ylabel("Outcome")
        ax = plt.gca()
        ax.set_facecolor('#efeeea')
        plt.bar(range(len(self.wins)), self.wins, color="#0A047A", width=1.0)
        plt.show()
        return