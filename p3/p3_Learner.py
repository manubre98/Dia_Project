import numpy as np

np.seterr(all='raise')


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.poisson_vector = np.zeros(shape=(n_arms, 2))

    def update_observations(self, pulled_arm, reward, _margin):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, _margin * reward)

    def update_poisson(self, return_info):
        mean, num = self.poisson_vector[return_info['arm'], 0], self.poisson_vector[return_info['arm'], 1]
        try:
            mean = (mean * num + return_info['average_returns'] * return_info['sample']) / (num + return_info['sample'])
        except (FloatingPointError or ZeroDivisionError):
            mean, num = 0, 0
        num = num + return_info['sample']
        self.poisson_vector[return_info['arm']] = np.array([mean, num])

    def update_poisson_context(self, return_info):
        num = self.poisson_vector[return_info['arm'], 1]
        mean = self.poisson_vector[return_info['arm'], 0]
        addmean, addnum = 0, 0
        for k in range(len(return_info['average_returns'])):
            addmean += return_info['average_returns'][k]*return_info['sample'][k]
            addnum += return_info['sample'][k]
        try:
            mean = (mean * num + addmean) / (num + addnum)
        except (FloatingPointError or ZeroDivisionError):
            mean, num = 0, 0
        num = num + addnum
        self.poisson_vector[return_info['arm']] = np.array([mean, num])
