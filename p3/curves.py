import numpy as np


class customer_class():
    def __init__(self, crspeed, crshift, poisson, maxclick, name=None):
        self.crspeed = crspeed
        self.crshift = crshift
        self.poisson = poisson
        self.maxclick = maxclick
        self.accesses = None
        self.poissons = None
        self.rates = None
        self.name = name
        self.index = None

    def conversion_rate(self, price):
        return 1 - 1 / (1 + np.exp(- self.crspeed * (price - self.crshift)))

    def returns(self):
        return self.poisson

    def clicks(self, bid):
        return self.maxclick * np.exp(10 * (bid - 0.75)) / (1 + np.exp(10 * (bid - 0.75)))

    def evaluate(self, bid, prices):
        self.accesses = np.round(self.clicks(bid))
        self.poissons = self.poisson
        self.rates = self.conversion_rate(prices)


def cost_per_click(bid):
    return (bid + 1) ** (4 / 5) - 1


def margin(price):
    return price - 4
