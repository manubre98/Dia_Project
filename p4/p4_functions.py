from p3.p3_TS_Learner import *
from p3.user_classes import *

from collections import deque
import numpy as np


def context_split(days, current_learner, classes, env, ts_dequy=None, previous_arms = []):
    return_ts = [current_learner]
    return_ts_classes = classes
    return_ts_dequy = [ts_dequy]
    for c in classes:
        c.evaluate(env.bid, env.prices)
    accesses = [c.accesses for c in classes]
    split = None

    # Create Learner
    current_rewards = []

    # generate empty deque
    if len(classes) == 4:
        ts_dequy = deque({'arm': 0, 'average_returns': np.zeros(shape=len(classes)),
                          'sample': np.zeros(shape=len(classes))} for _ in range(env.return_time))

    arms = []
    d = 0
    # Simulate experiments
    for d in range(days):

        # choose arm
        ts_pulled_arm = current_learner.pull_arm(env.margins)
        arms.append(ts_pulled_arm)

        # simulate accesses
        # empty daily reward
        ts_daily = {'reward': 0, 'successes': 0, 'returns': 0}
        clicks, purchases, returns = [], [], []
        ts_dicty = {'arm': ts_pulled_arm, 'average_returns': np.zeros(shape=len(classes)),
                    'sample': np.zeros(shape=len(classes))}

        for i, c in enumerate(classes):
            n_trials = int(c.accesses)
            # quanti acquisti
            ts_successes, ts_reward, ts_class_returns = env.round(ts_pulled_arm, c, n_trials)
            # aggiorno beta
            current_learner.update(ts_pulled_arm, ts_successes, n_trials)

            # aggiorno reward_giornaliera
            try:
                avi = ts_class_returns / ts_successes
            except ZeroDivisionError:
                avi = 0
            ts_dicty['average_returns'][i] = avi
            ts_dicty['sample'][i] = ts_successes
            ts_daily['reward'] += ts_reward
            ts_daily['successes'] += ts_successes
            ts_daily['returns'] += ts_class_returns

            clicks.append(n_trials)
            purchases.append(ts_successes)
            returns.append(ts_class_returns)

        if len(clicks) == 4:
            r0 = Register(pulled_arm=ts_pulled_arm, _class=0, clicks=clicks[0],
                          purchases=purchases[0], returns=returns[0])
            r1 = Register(pulled_arm=ts_pulled_arm, _class=1, clicks=clicks[1],
                          purchases=purchases[1], returns=returns[1])
            r2 = Register(pulled_arm=ts_pulled_arm, _class=2, clicks=clicks[2],
                          purchases=purchases[2], returns=returns[2])
            r3 = Register(pulled_arm=ts_pulled_arm, _class=3, clicks=clicks[3],
                          purchases=purchases[3], returns=returns[3])

            current_learner.table.update([r0, r1, r2, r3])

        elif len(clicks) == 2:
            r0 = Register(pulled_arm=ts_pulled_arm, _class=0, clicks=clicks[0],
                          purchases=purchases[0], returns=returns[0])
            r1 = Register(pulled_arm=ts_pulled_arm, _class=1, clicks=clicks[1],
                          purchases=purchases[1], returns=returns[1])
            current_learner.table.update([r0, r1])

        # work on dequy
        ts_dequy.append(ts_dicty)
        current_learner.update_poisson_context(ts_dequy.popleft())

        # save daily reward
        current_rewards.append(ts_daily['reward'])

        # if convergenza: valuta split
        split = None
        arg_list = ts_pulled_arm
        if check_convergence(arms=arms)[0] and current_learner.table.n_classes != 0:
            split, arg_list = current_learner.evaluate_means(env.margins, chosen_arms=previous_arms+arms)
            if int(len(classes) / 2) != split:
                print('Number of features: ', int(len(classes) / 2), 'Proposed split: ', split, 'Proposed prices',
                  env.prices[arg_list])

        # if split: crea nuovi thompson, return_ts = [th1, th2], return_ts_classes = [classi1, classi2], esci dal ciclo
        if current_learner.table.n_classes == 2 and split == 0:
            class0 = classes[0]
            class1 = classes[1]
            ts0 = TS_Learner(n_arms=env.n_arms, names=[class0.name], n_features=0)
            ts0.inherit(current_learner, split, 0)
            ts0_dequy = adapt_dequy(ts_dequy, [0])
            ts1 = TS_Learner(n_arms=env.n_arms, names=[class1.name], n_features=0)
            ts1.inherit(current_learner, split, 1)
            ts1_dequy = adapt_dequy(ts_dequy, [1])
            return_ts = [ts0, ts1]
            return_ts_classes = [[class0], [class1]]
            return_ts_dequy = [ts0_dequy, ts1_dequy]
            break

        if current_learner.table.n_classes == 4 and split != 2 and split is not None:
            class0, class1 = None, None
            ts0_dequy, ts1_dequy = None, None
            if split == 0:
                class0, class1 = [classes[0], classes[2]], [classes[1], classes[3]]
                ts0_dequy, ts1_dequy = adapt_dequy(ts_dequy, [0, 2]), adapt_dequy(ts_dequy, [1, 3])
            if split == 1:
                class0, class1 = [classes[0], classes[1]], [classes[2], classes[3]]
                ts0_dequy, ts1_dequy = adapt_dequy(ts_dequy, [0, 1]), adapt_dequy(ts_dequy, [2, 3])
            ts0 = TS_Learner(n_arms=env.n_arms, names=[class0[0].name, class0[1].name], n_features=1)
            ts0.inherit(current_learner, split, 0)
            ts1 = TS_Learner(n_arms=env.n_arms, names=[class1[0].name, class1[1].name], n_features=1)
            ts1.inherit(current_learner, split, 1)
            return_ts = [ts0, ts1]
            return_ts_classes = [class0, class1]
            return_ts_dequy = [ts0_dequy, ts1_dequy]
            break

    # esci dalla funzione ritornando return_ts, return_ts_classes, giorni_mancanti, reward ottenuta fino ad ora
    #arms = previous_arms + arms
    return return_ts, return_ts_classes, return_ts_dequy, current_rewards, days - d - 1, split, arms, env.prices[arg_list].tolist()


def check_convergence(arms, thres=400, crit=0.7):
    conv = False
    unique_elements, count_elements = np.unique(np.array(arms[-int(0.10 * thres):]), return_counts=True)
    if len(arms) >= 0.10 * thres and np.max(count_elements) > crit * len(arms[-int(0.10 * thres):]):
        conv = True
    best_arm = np.bincount(arms[-int(0.10 * thres):]).argmax()
    return conv, best_arm


def adapt_dequy(dequy, indexes):
    new_dequy = deque()
    for dicty in dequy:
        new_dicty = {'arm': dicty['arm'], 'average_returns': dicty['average_returns'][indexes],
                     'sample': dicty['sample'][indexes]}
        new_dequy.append(new_dicty)
    return new_dequy


def compute_optimum_pricing(prices, bid, user_classes):
    opt_vec = []
    for i in range(np.size(prices)):
        val = 0
        for c in user_classes:
            val += c.clicks(bid) * (
                        c.conversion_rate(prices[i]) * margin(prices[i]) * (c.returns() + 1) - cost_per_click(bid))
        opt_vec.append(val)

    opt_ind = np.argmax(np.array(opt_vec))
    opt = np.max(np.array(opt_vec))

    return opt, opt_ind, opt_vec


def aggregate_rewards(rewards0, rewards1, rewards2, T):
    agg_rew = np.zeros(shape=T)
    d = 0
    start = len(rewards0)
    agg_rew[:start] += rewards0
    for rew in rewards1:
        middle = len(rew)
        agg_rew[start:start + middle] += rew

    for rew2 in rewards2:
        end = len(rew2)
        agg_rew[-end:] += rew2

    return agg_rew
