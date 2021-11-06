import copy

import numpy as np
import matplotlib.pyplot as plt


def plot_regret(opt, final_rewards, names=None, color_list='r'):
    zippy = zip(final_rewards, color_list)
    plt.figure(0)
    plt.title("REGRET - {} simulations".format(len(final_rewards[0])))
    plt.xlabel("t")
    plt.ylabel("regret")
    leg = names
    maxes = []
    handles = []
    for line, col in zippy:
        curve = np.nancumsum(np.nanmean(opt - line, axis=0))
        maxes.append(np.max(curve))
        plt.plot(curve, col)
        for i in range(len(final_rewards[0])):
            plt.plot(np.nancumsum(opt - line[i], axis=0), 'g', alpha=1 / np.power(len(final_rewards[0]), 2 / 3))

    plt.ylim([0, 1.8 * np.max(maxes, axis=0)])
    for ind, han in enumerate(handles):
        han.set_color(color_list[ind])
    plt.legend(leg) #handles=handles)
    plt.show()

def plot_regret2(opt, final_rewards, names=None, color_list='r'):
    zippy = zip(final_rewards, color_list)
    plt.figure(0)
    plt.title("REGRET - {} simulations".format(len(final_rewards[0])))
    plt.xlabel("t")
    plt.ylabel("regret")
    leg = names
    maxes = []
    handles = []
    curve = [None, None]

    curve[0] = np.nancumsum(np.nanmean(opt - final_rewards[0], axis=0))
    maxes.append(np.max(curve[0]))
    plt.plot(curve[0], color_list[0])

    curve[1] = np.nancumsum(np.nanmean(opt - final_rewards[1], axis=0))
    maxes.append(np.max(curve[1]))
    plt.plot(curve[1], color_list[1])

    for line, col in zippy:
        for i in range(len(final_rewards[0])):
            plt.plot(np.nancumsum(opt - line[i], axis=0), col, alpha=1 / np.power(len(final_rewards[0]), 2 / 3))

    plt.ylim([0, 1.8 * np.max(maxes, axis=0)])
    for ind, han in enumerate(handles):
        han.set_color(color_list[ind])
    plt.legend(leg) #handles=handles)
    plt.show()


def plot_reward(opt, final_rewards, names=None, color_list='r', optbest = None):
    zippy = zip(final_rewards, color_list)
    plt.figure(0)
    plt.title("REWARD - {} simulations".format(len(final_rewards[0])))
    plt.xlabel("t")
    plt.ylabel("reward")
    leg = names
    leg.insert(0, 'Disaggregate Optimum')
    if optbest is not None: leg.insert(0, 'Aggregate Optimum')
    maxes = []
    opt_line = np.ones(shape=len(final_rewards[0][0])) * opt
    if optbest is not None: best_line = np.ones(shape=len(final_rewards[0][0])) * optbest
    plt.plot(opt_line, 'k')
    if optbest is not None: plt.plot(best_line, 'b')
    for line, col in zippy:
        curve = np.nanmean(line, axis=0)
        maxes.append(np.max(curve))
        plt.plot(curve, col)
        for i in range(len(final_rewards[0])):
            plt.plot(line[i], 'g', alpha=1 / np.power(len(final_rewards[0]), 1))

    plt.ylim([16000, 1.1 * np.max(maxes, axis=0)])
    plt.legend(leg)
    plt.show()

def plot_reward2(opt, final_rewards, names=None, color_list='r', optbest = None):
    zippy = zip(final_rewards, color_list)
    plt.figure(0)
    plt.title("REWARD - {} simulations".format(len(final_rewards[0])))
    plt.xlabel("t")
    plt.ylabel("reward")
    leg = names
    leg.insert(0, 'Disaggregate Optimum')
    if optbest is not None: leg.insert(0, 'Aggregate Optimum')
    maxes = []
    curve = [None, None]

    opt_line = np.ones(shape=len(final_rewards[0][0])) * opt
    if optbest is not None: best_line = np.ones(shape=len(final_rewards[0][0])) * optbest
    plt.plot(opt_line, 'k')
    if optbest is not None: plt.plot(best_line, 'b')

    curve[0] = np.nanmean(final_rewards[0], axis=0)
    maxes.append(np.max(curve[0]))
    plt.plot(curve[0], color_list[0])

    curve[1] = np.nanmean(final_rewards[1], axis=0)
    maxes.append(np.max(curve[1]))
    plt.plot(curve[1], color_list[1])

    for line, col in zippy:


        for i in range(len(final_rewards[0])):
            plt.plot(line[i], col, alpha=1 / np.power(len(final_rewards[0]), 1))

    plt.ylim([16000, 1.1 * np.max(maxes, axis=0)])
    plt.legend(leg)
    plt.show()


def double_nested_loop(list_of_list):
    return_thing = []
    for ind1 in list_of_list:
        for ind2 in ind1:
            return_thing.append(ind2)
    return return_thing
