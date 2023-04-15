import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, env, holes=None, frozen=None) -> None:
        self.env = env
        self.q_table = np.zeros([self.env.observation_space.n,
                                 self.env.action_space.n])
        self.frames = []
        self.eval_frames = []
        self.holes = holes
        self.frozen = frozen
        self.rewards = None
        self.wins = None

    def q_train(self, alpha, gamma, epsilon, episodes=1000, modify_rewards=False, save_images=False):

        self.rewards = np.zeros(episodes)
        self.wins = np.zeros(episodes)

        for i in tqdm(range(episodes)):
            state = self.env.reset()[0]
            reward = 0
            done = False
            
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values

                next_state, reward, done, _, _ = self.env.step(action)

                if modify_rewards:
                    if next_state in self.holes:
                        reward = reward - 0.2
                    
                    elif next_state in self.frozen:
                        reward = reward - 0.01
                
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value
                self.rewards[i] += reward

                if reward == 1:
                    self.wins[i] += 1

                state = next_state
                img = self.env.render()
                
                if save_images:
                    self.frames.append(img)
    
    def evaluate_performance(self, episodes=100, save_images=False):
        total_epochs, total_penalties = 0, 0

        for i in tqdm(range(episodes)):
            state = self.env.reset()[0]
            epochs, penalties, reward = 0, 0, 0
            
            done = False
            
            while not done:
                action = np.argmax(self.q_table[state])
                next_state, reward, done, truncated, info = self.env.step(action)

                if next_state in self.holes:
                    penalties += 1

                state = next_state
                img = self.env.render()
                if save_images:
                    self.eval_frames.append(img)

                epochs += 1

            total_penalties += penalties
            total_epochs += epochs
        
        return total_epochs, total_penalties

    def plot_q_training_curve(self, window=10):
        avg_rewards = np.array([np.mean(self.rewards[i-window:i])  
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

    def plot_q_training_wins(self):
        plt.figure(figsize=(12, 5))
        plt.xlabel("Run number")
        plt.ylabel("Outcome")
        ax = plt.gca()
        ax.set_facecolor('#efeeea')
        plt.bar(range(len(self.wins)), self.wins, color="#0A047A", width=1.0)
        plt.show()



        
        