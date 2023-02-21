#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/1/22 下午12:10:22
# @Author : ZZZCY
# @File : [Hands-On-RL] --> MonteCarlo.py

import numpy as np


def join(str1, str2):
    return str1 + '-' + str2


def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, GAMMA = MDP  # 拆分数据
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机抽取S1-S4的状态作为起点
        while s != "S5" and timestep <= timestep_max:  # S5为终点，步数不超过最大范围
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:  # 遍历所有动作
                temp += Pi.get(join(s, a_opt), 0)  # 在S状态下采取动作a_opt的概率
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率获取下一状态
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes


def MC(episodes, V, N, GAMMA):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  # 从大到小排序
            (s, a, r, s_next) = episode[i]  # 拆包
            G = r + GAMMA * G  # 累计奖励
            N[s] = N[s] + 1  # 计数值增加
            V[s] = V[s] + (G - V[s]) / N[s]  # 价值更新


np.random.seed(0)

S = ["S1", "S2", "S3", "S4", "S5"]  # 状态
A = ["KEEP S1", "GO S1", "GO S2", "GO S3", "GO S4", "GO S5", "MAY GO"]  # 动作
P = {
    "S1-KEEP S1-S1": 1.0, "S1-GO S2-S2": 1.0,
    "S2-GO S1-S1": 1.0, "S2-GO S3-S3": 1.0,
    "S3-GO S4-S4": 1.0, "S3-GO S5-S5": 1.0,
    "S4-GO S5-S5": 1.0, "S4-MAY GO-S2": 0.2, "S4-MAY GO-S3": 0.4, "S4-MAY GO-S4": 0.4
}  # 状态转移函数
R = {
    "S1-KEEP S1": -1, "S1-GO S2": 0,
    "S2-GO S1": -1, "S2-GO S3": -2,
    "S3-GO S4": -2, "S3-GO S5": 0,
    "S4-GO S5": 10, "S4-MAY GO": 1
}  # 奖励函数
GAMMA = 0.5  # 折扣因子
MDP = (S, A, P, R, GAMMA)
PI_1 = {
    "S1-KEEP S1": 0.5, "S1-GO S2": 0.5,
    "S2-GO S1": 0.5, "S2-GO S3": 0.5,
    "S3-GO S4": 0.5, "S3-GO S5": 0.5,
    "S4-GO S5": 0.5, "S4-MAY GO": 0.5
}  # 策略1

# episodes = sample(MDP, PI_1, 20, 5)
# print("第1条序列\n", episodes[0])
# print("第2条序列\n", episodes[1])
# print("第5条序列\n", episodes[4])

timestep_max = 20
episodes = sample(MDP, PI_1, timestep_max, 1000)
GAMMA = 0.5
V = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0}  # 状态价值
N = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0}  # 计数器
MC(episodes, V, N, GAMMA)
print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)
