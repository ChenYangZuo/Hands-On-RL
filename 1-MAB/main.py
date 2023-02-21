import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)  # 定义K根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆尝试次数
        self.regret = 0.  # 当前步累积懊悔
        self.actions = []  # 维护列表，记录动作
        self.regrets = []  # 维护列表，记录累积懊悔

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError  # 相当于C++的虚函数，若没有被重写就运行则报错

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


# Epsilon-贪婪算法
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化所有拉杆的期望奖励值

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随即选择拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望最大的拉杆
        r = self.bandit.step(k)  # 获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


# 时变Epsilon-贪婪算法
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0  # 运行次数

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


# 上置信界算法
class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


# 汤普森采样算法
class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按Beta分布采样一次
        k = np.argmax(samples)  # 取奖励最大的动作
        r = self.bandit.step(k)
        self._a[k] += r
        self._b[k] += (1 - r)
        return k


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title(f"{solvers[0].bandit.K}-armed bandit")
    plt.legend()
    plt.show()


def epsilon_greedy_test():
    np.random.seed(1)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


def decaying_epsilon_greedy_test():
    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print(f"累积懊悔为{decaying_epsilon_greedy_solver.regret}")
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


def ucb_test():
    np.random.seed(1)
    coef = 1
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print(f"累积懊悔为{UCB_solver.regret}")
    plot_results([UCB_solver], ["UCB"])


def thompson_sampling_test():
    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print(f"累积懊悔为{thompson_sampling_solver.regret}")
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("-----------------init Start-----------------")
print(f"随机生成一个{K}臂伯努里老虎机")
print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}，获奖概率为{bandit_10_arm.best_prob}")
print("------------------init End------------------")

ucb_test()
