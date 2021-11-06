import numpy as np

from p3.user_classes import *


class BiddingEnvironment6:
    def __init__(self, bids, acc_sigma, cost_sigma, user_classes, n_arms):
        self.bids = bids
        self.acc_means = self.initialize_accesses(user_classes=user_classes, bids=bids)
        self.cost_means = self.initialize_cost(user_classes=user_classes, bids=bids)
        self.acc_sigmas = np.ones(len(bids)) * acc_sigma
        self.cost_sigmas = np.ones(len(bids)) * cost_sigma
        self.n_arms = n_arms

    def initialize_accesses(self, user_classes, bids):
        means = np.zeros(shape=(len(user_classes), len(bids)))
        for ii, c in enumerate(user_classes):
            for j, b in enumerate(bids):
                means[ii, j] = c.clicks(b)
        return means

    def initialize_cost(self, user_classes, bids):
        means = np.zeros(shape=(len(user_classes), len(bids)))
        for ii, c in enumerate(user_classes):
            for j, b in enumerate(bids):
                means[ii, j] = cost_per_click(b)
        return means

    def round(self, pulled_arm, user_c):
        sample_accesses = np.random.normal(self.acc_means[user_c, pulled_arm], self.acc_sigmas[pulled_arm])
        sample_cost = np.random.normal(self.cost_means[user_c, pulled_arm], self.cost_sigmas[pulled_arm])
        if sample_cost > self.bids[pulled_arm]:
            sample_cost = self.bids[pulled_arm]
        if sample_cost < 0:
            sample_cost = 0
        if int(sample_accesses) < 0:
            sample_accesses = self.acc_means[user_c, pulled_arm]

        return int(sample_accesses), sample_cost
