#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/2/27 下午6:46:48
# @Author : ZZZCY
# @File : [Hands-On-RL] --> Actor-Critic.py

import gym
import torch
import torch.cuda
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


# 策略网络，输入状态，输出动作
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


# 价值网络，输入状态，输出状态价值
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 生成演员和评论家
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 生成优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # 初始化参数
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # 生成state张量，float类型，运行在指定设备上
        probs = self.actor(state)  # 参数化的PolicyNet
        action_dist = torch.distributions.Categorical(probs)  # 输出action对应的概率
        action = action_dist.sample()  # 对动作进行采样
        return action.item()

    def update(self, transition_dict):
        # 解包
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 时序差分的目标
        td_delta = td_target - self.critic(states)  # 时序差分的误差
        log_probs = torch.log(self.actor(states).gather(1, actions))  # 映射后对数运算
        """
        detach():
        返回一个新的tensor，从当前计算图中分离下来。但是仍指向原变量的存放位置，不同之处只是requirse_grad为false.得到的这个tensir永远不需要计算器梯度，不具有grad.
        即使之后重新将它的requires_grad置为true,它也不会具有梯度grad.这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播.
        """
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # 梯度置零
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # 计算梯度
        actor_loss.backward()
        critic_loss.backward()
        # 更新网络参数
        self.actor_optimizer.step()
        self.critic_optimizer.step()


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Actor-Critic on {}".format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Actor-Critic on {}".format(env_name))
plt.show()
