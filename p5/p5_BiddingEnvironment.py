from p3.user_classes import *


def fun(c, b, p, sigmacost = 0):
    sample_cost = np.random.normal(cost_per_click(b), sigmacost)
    if sample_cost > b:
        sample_cost = b
    if sample_cost < 0:
        sample_cost = 0
    obj = c.clicks(b) * (c.conversion_rate(p) * margin(p) * (c.poisson + 1)) - c.clicks(b) * cost_per_click(b)    #sample_cost
    return obj


class BiddingEnvironment:
    def __init__(self, bids, sigma, user_classes, price, n_arms):
        self.bids = bids
        self.means = self.initialize_means(user_classes=user_classes, bids=bids, price=price)
        self.sigmas = np.ones(len(bids)) * sigma
        self.price = price
        self.n_arms = n_arms

    def initialize_means(self, user_classes, bids, price):
        means = np.zeros(shape=(len(user_classes), len(bids)))
        for ii, c in enumerate(user_classes):
            for j, b in enumerate(bids):
                means[ii, j] = fun(c, b, price)
        return means

    def round(self, pulled_arm, user_c):
        c = classes[user_c]
        p = self.price
        b = self.bids[pulled_arm]
        mean_ = fun(c,b,p, sigmacost = 0)
        val = (c.conversion_rate(p) * margin(p) * (c.poisson + 1)) -  cost_per_click(b)
        variance = np.power(self.sigmas[pulled_arm],2)*0.01+np.power(self.sigmas[pulled_arm],2)*np.power(val,2) + 0.01 * np.power(c.clicks(b),2)
        #return np.random.normal(mean_, np.sqrt(variance))
        return np.random.normal(self.means[user_c, pulled_arm], self.sigmas[pulled_arm])
