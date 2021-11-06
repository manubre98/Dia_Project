from p3.user_classes import *


class PricingEnvironment6:
    def __init__(self, n_arms, probabilities, margins, poissons, return_time=None, prices=None):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.margins = margins
        self.poissons = np.array(poissons)
        self.return_time = return_time
        self.prices = prices

    def round(self, pulled_arm, user_c, n_trials=1):
        successes = np.random.binomial(n_trials, self.probabilities[user_c.index][pulled_arm])
        number_returns = np.random.poisson(successes * self.poissons[user_c.index])
        marg = self.margins[pulled_arm] * (number_returns + successes)
        return successes, number_returns
