#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/1/24 下午5:31:39
# @Author : ZZZCY
# @File : [Hands-On-RL] --> CliffWalking.py

import copy


class CliffWalkingEnv:
    """悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 网格世界的列
        self.nrow = nrow  # 网格世界的行
        self.P = self.create_p()
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]
        # print(self.P)

    def create_p(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4 actions: change[0]:up, change[1]:down, change[2]:left, change[3]:right
        # 坐标系原点在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 当前在悬崖或目的地
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_s = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或目的地
                    if next_y == self.nrow - 1 and next_x > 0:  # 目的地
                        done = True
                        if next_x != self.ncol - 1:  # 悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_s, reward, done)]
        return P


class PolicyIteration:
    """策略迭代算法"""

    def __init__(self, env: CliffWalkingEnv, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    # 策略评估
    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0  # 最大差距
            new_v = [0] * self.env.ncol * self.env.nrow  # 创建新状态价值
            for s in range(self.env.ncol * self.env.nrow):  # 遍历所有状态
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)
                for a in range(4):  # 4种动作
                    qsa = 0
                    for res in self.env.P[s][a]:  # 在state下采取action会出现：
                        p, next_state, r, done = res  # 拆包
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:  # 策略评估收敛
                break
            cnt += 1
        print(f"策略评估进行{cnt}轮后完成")

    # 策略提升
    def policy_improvement(self):
        for s in range(self.env.ncol * self.env.nrow):  # 遍历所有状态
            qsa_list = []
            for a in range(4):  # 遍历所有动作
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)  # 获取动作的最大Q值
            cntq = qsa_list.count(maxq)  # 有cntq个动作获得最大Q值
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]  # 让策略在s状态时均匀随机使用取最大Q值的动作
        print("策略提升完成")
        return self.pi

    # 策略迭代
    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 深拷贝
            new_pi = self.policy_improvement()
            if old_pi == new_pi:  # 策略不再改变
                break


class ValueIteration:
    """价值迭代算法"""

    def __init__(self, env: CliffWalkingEnv, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    # 价值迭代
    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print(f"价值迭代共进行{cnt}轮")
        self.get_policy()

    # 根据价值函数到处贪婪策略
    def get_policy(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
                maxq = max(qsa_list)
                cntq = qsa_list.count(maxq)
                self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print("%6.6s" % ("%.3f" % agent.v[i * agent.env.ncol + j]), end=" ")
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:  # 悬崖
                print("****", end=" ")
            elif (i * agent.env.ncol + j) in end:  # 终点
                print("EEEE", end=" ")
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else "o"
                print(pi_str, end=" ")
        print()


m_env = CliffWalkingEnv()
action_meaning = ["^", "v", "<", ">"]
theta = 0.001
gamma = 0.9

# 方法1：策略迭代
# agent = PolicyIteration(m_env, theta, gamma)
# agent.policy_iteration()
# 方法2：价值迭代
agent = ValueIteration(m_env, theta, gamma)
agent.value_iteration()

# 打印Agent
print_agent(agent, action_meaning, list(range(37, 47)), [47])
