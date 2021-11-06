import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from p3.p3_Learner import *
from utils import *

class GPTS_Learner6(Learner):
    def __init__(self, n_arms, arms, n_features=2, price=None, names=None):
        super().__init__(n_arms)
        self.arms = arms
        self.acc_means = np.zeros(self.n_arms)
        self.acc_sigmas = np.ones(self.n_arms)
        self.cost_means = np.zeros(self.n_arms)
        self.cost_sigmas = np.ones(self.n_arms)
        self.pulled_arms = []
        self.collected_clicks = []
        self.collected_costs = []
        alpha_acc = 1000
        kernel_acc = C(100, (100, 1e6)) * RBF(10, (1e-1, 1e6))
        self.gp_acc = GaussianProcessRegressor(kernel=kernel_acc, alpha=alpha_acc, normalize_y=False, n_restarts_optimizer=1)

        alpha_cost = 0.3
        kernel_cost = C(0.1, (1, 1e2)) * RBF(0.1, (1, 1e2))
        self.gp_cost = GaussianProcessRegressor(kernel=kernel_cost, alpha=alpha_cost, normalize_y=False, n_restarts_optimizer=1)

        # self.table = SplitTable(n_arms, n_classes=n_features * 2, names=names)
        self.names = names
        self.price = price

    def update_observations_gpts(self, pulled_arm, clicks, costs):
        # self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_clicks = np.append(self.collected_clicks, clicks)
        self.collected_costs = np.append(self.collected_costs, costs)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_clicks
        self.gp_acc.fit(x, y)
        self.acc_means, self.acc_sigmas = self.gp_acc.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.acc_sigmas = np.maximum(self.acc_sigmas, 30)

        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_costs
        self.gp_cost.fit(x, y)
        self.cost_means, self.cost_sigmas = self.gp_cost.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.cost_sigmas = np.maximum(self.cost_sigmas, 0.01)

    def update(self, pulled_arm, clicks, costs):
        self.t += 1
        self.update_observations_gpts(pulled_arm, clicks, costs)
        self.update_model()

    def pull_arm(self, pricing_learner, price_idx, margins):
        if self.t < self.n_arms:
            return self.t  # % self.n_arms

        try:
            conv_rate = pricing_learner.beta_parameters[price_idx, 0] / (pricing_learner.beta_parameters[price_idx, 0]
                                                                         + pricing_learner.beta_parameters[
                                                                             price_idx, 1])
        except ZeroDivisionError:
            conv_rate = 0
            print('DIV 0')
        margine = margins[price_idx]
        poisson = pricing_learner.poisson_vector[price_idx, 0] + 1

        exp_rew = np.random.normal(self.acc_means * (np.ones(shape=self.n_arms)
                                                     * margine * conv_rate * poisson - self.cost_means), 50)
        bid_idx = np.argmax(exp_rew)

        return bid_idx

    def pull_arm_context(self, learners_list, price_idx_list, margins, classes_list):

        classes_tot = double_nested_loop(classes_list)
        accesses_tot_l = [c.accesses for c in classes_tot]
        accesses_tot = np.sum(np.array(accesses_tot_l))

        if self.t < self.n_arms:
            return self.t  # % self.n_arms

        pricing_term = 0
        coefficient = np.zeros(shape=len(classes_list))
        for index, pricing_learner in enumerate(learners_list):
            try:
                conv_rate = pricing_learner.beta_parameters[price_idx_list[index], 0] / (pricing_learner.beta_parameters[price_idx_list[index], 0]
                                                                            + pricing_learner.beta_parameters[
                                                                                price_idx_list[index], 1])
            except ZeroDivisionError:
                conv_rate = 0
                print('DIV 0')
            margine = margins[index]
            poisson = pricing_learner.poisson_vector[price_idx_list[index], 0] + 1
            acc_this = [c.accesses for c in classes_list[index]]
            coefficient[index] = np.sum(np.array(acc_this))/(accesses_tot)

            pricing_term += coefficient[index] * margine * conv_rate * poisson

        exp_rew = np.random.normal(self.acc_means * (np.ones(shape=self.n_arms)
                                                     * pricing_term - self.cost_means), 50)
        bid_idx = np.argmax(exp_rew)

        return bid_idx
