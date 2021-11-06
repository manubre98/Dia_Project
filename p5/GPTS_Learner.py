from p3.p3_TS_Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W


class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms, n_features=2, price=None, names=None):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 1e6
        self.pulled_arms = []
        alpha = 1000
        kernel = C(100, (100, 1e6)) * RBF(1, (1e-1, 1e1))
        #kernel = W(1000**2) + RBF(1, (1e-1, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=False, n_restarts_optimizer=1)

        #self.table = SplitTable(n_arms, n_classes=n_features * 2, names=names)
        self.names = names
        self.price = price

    def update_observations_gpts(self, pulled_arm, reward):
        #self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 30)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations_gpts(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t  # % self.n_arms
        s_val = self.means
        for i in range(len(s_val)):
            if s_val[i] < 2000 and self.t > 20:
                s_val[i] = 0
                print('ok')
        sampled_values = np.random.normal(s_val, self.sigmas)
        return np.argmax(sampled_values)

