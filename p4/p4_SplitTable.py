from __future__ import print_function
from pandas import *

import numpy as np


class Register:
    def __init__(self, pulled_arm, _class, clicks, purchases, returns, names=None):
        self.arm = pulled_arm
        self.register = {'class': _class, 'clicks': clicks, 'purchases': purchases, 'returns': returns}
        self.names = names


class SplitTableDEPRECATED:
    def __init__(self, n_arms, n_features=2, n_categories=2, n_infos=3, names=None):
        self.matrix = np.zeros(shape=(n_arms, n_features, n_categories, n_infos), dtype=int)
        self.names = names
        self.n_features = n_features

    def update(self, update_list):
        for index, feature_update in enumerate(update_list):
            for dic in feature_update.register:
                self.matrix[feature_update.arm, index, dic['value'], 0] += dic['clicks']
                self.matrix[feature_update.arm, index, dic['value'], 1] += dic['purchases']
                self.matrix[feature_update.arm, index, dic['value'], 2] += dic['returns']

    def print(self, prices):
        shape = self.matrix.shape
        print(shape)
        for row in range(shape[0]):
            print('\n', 'Arm: ', row, 'Price: ', "{:.3f}".format(prices[row]), '\n')
            for feature in range(shape[1]):
                df = DataFrame(self.matrix[row, feature, :, :])
                df.rename(columns={0: 'Clicks', 1: 'Purchases', 2: 'Returns'}, inplace=True)
                df.rename(index={0: self.names[feature][0], 1: self.names[feature][1]}, inplace=True)
                print(df)


class SplitTable:
    def __init__(self, n_arms, n_classes=4, n_infos=3, names=None):
        self.matrix = np.zeros(shape=(n_arms, n_classes, n_infos), dtype=int)
        self.names = names
        self.n_classes = n_classes

    def update(self, update_list):
        for r in update_list:
            self.matrix[r.arm, r.register['class'], 0] += r.register['clicks']
            self.matrix[r.arm, r.register['class'], 1] += r.register['purchases']
            self.matrix[r.arm, r.register['class'], 2] += r.register['returns']

    def print(self, prices):
        shape = self.matrix.shape
        print(shape)
        for row in range(shape[0]):
            print('\n', 'Arm: ', row, 'Price: ', "{:.3f}".format(prices[row]), '\n')
            df = DataFrame(self.matrix[row, :, :])
            df.rename(columns={0: 'Clicks', 1: 'Purchases', 2: 'Returns'}, inplace=True)
            #df.rename(index={0: 'Clicks', 1: 'Purchase', 2: 'Returns'}, inplace=True)
            print(df)
