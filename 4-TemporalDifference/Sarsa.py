#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/2/21 下午6:43:32
# @Author : ZZZCY
# @File : [Hands-On-RL] --> Sarsa.py

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 当前智能体横坐标
        self.y = self.nrow - 1  # 当前智能体纵坐标

    def step(self, action):
        # 4 actions: change[0]:up, change[1]:down, change[2]:left, change[3]:right
        # 坐标系原点在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.nrow + self.x


class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 48行 4列
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # epsilon-贪婪算法
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:  # 若两个动作的价值一样，都会记录下来
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    print("策略：")
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:  # 悬崖
                print("****", end=" ")
            elif (i * env.ncol + j) in end:  # 终点
                print("EEEE", end=" ")
            else:
                a = agent.best_action([i * env.ncol + j])
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else "o"
                print(pi_str, end=" ")
        print()


ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500

return_list = []
for i in range(10):  # 显示10个进度条
    with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({"episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                                  "return": "%.3f" % np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Sarsa on CliffWalking")
plt.show()

action_meaning = ["^", "v", "<", ">"]
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
