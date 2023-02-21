#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/1/22 下午12:10:22
# @Author : ZZZCY
# @File : [Hands-On-RL] --> BellmanExpectation.py

import numpy as np


# 计算价值函数
# 需要进行矩阵求逆和矩阵乘法，适用于小范围求解
def compute(p, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1))  # 转换为列向量
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * p), rewards)
    # np.eye 对角矩阵
    # np.linalg.inv 逆矩阵
    # np.dot 矩阵乘法
    return value


S = ["S1", "S2", "S3", "S4", "S5"]  # 状态
A = ["KEEP S1", "GO S1", "GO S2", "GO S3", "GO S4", "GO S5", "MAY GO"]  # 动作
P = {
    "S1-KEEP S1-S1": 1.0, "S1-GO S2-S2": 1.0,
    "S2-GO S1-S1": 1.0, "S2-GO S3-S3": 1.0,
    "S3-GO S4-S4": 1.0, "S3-GO S5-S5": 1.0,
    "S4-GO S5-S5": 1.0, "S4-MAY-S2": 0.2, "S4-MAY-S3": 0.4, "S4-MAY-S4": 0.4
}  # 状态转移函数
R = {
    "S1-KEEP S1": -1, "S1-GO S2": 0,
    "S2-GO S1": -1, "S2-GO S3": -2,
    "S3-GO S4": -2, "S3-GO S5": 0,
    "S4-GO S5": 10, "S4-MAY": 1
}  # 奖励函数
GAMMA = 0.5  # 折扣因子
MDP = (S, A, P, R, GAMMA)

PI_1 = {
    "S1-KEEP S1": 0.5, "S1-GO S2": 0.5,
    "S2-GO S1": 0.5, "S2-GO S3": 0.5,
    "S3-GO S4": 0.5, "S3-GO S5": 0.5,
    "S4-GO S5": 0.5, "S4-MAY": 0.5
}  # 策略1

PI_2 = {
    "S1-KEEP S1": 0.6, "S1-GO S2": 0.4,
    "S2-GO S1": 0.3, "S2-GO S3": 0.7,
    "S3-GO S4": 0.5, "S3-GO S5": 0.5,
    "S4-GO S5": 0.1, "S4-MAY": 0.9
}  # 策略2

P1_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0]
]

P1_from_mdp_to_mrp = np.array(P1_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P1_from_mdp_to_mrp, R_from_mdp_to_mrp, GAMMA, 5)
print("MDP中每个状态价值分别为\n", V)
