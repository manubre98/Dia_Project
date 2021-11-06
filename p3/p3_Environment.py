from .user_classes import *


def fun(b, marg, n_trials):
    obj = marg - n_trials * cost_per_click(b)
    return obj


class Environment:
    def __init__(self, n_arms, probabilities, margins, poissons, return_time=None, prices=None, bid=None):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.margins = margins
        self.poissons = np.array(poissons)
        self.return_time = return_time
        self.prices = prices
        self.bid = bid

    def round(self, pulled_arm, user_c, n_trials=1):
        successes = np.random.binomial(n_trials, self.probabilities[user_c.index][pulled_arm])
        number_returns = np.random.poisson(successes * self.poissons[user_c.index])
        marg = self.margins[pulled_arm] * (number_returns + successes)
        obj = fun(b=self.bid, marg=marg, n_trials=n_trials)
        return successes, obj, number_returns
