import numpy as np

from p3.p3_Environment import *
from p4.p4_functions import *
from utils import *
from tqdm import tqdm

np.random.seed(898)
# Environment
n_arms = 10
prices = np.linspace(5, 14, n_arms)
margins = margin(prices)
T = 365
return_time = 30
table = None

# People
names = ['Young ', 'Old', 'Not Sporty', 'Sporty']
bid = 1.2
accesses = []
poissons = []
rates = []
for c in classes:
    accesses.append(np.round(c.clicks(bid)))
    poissons.append(c.poisson)
    rates.append(c.conversion_rate(prices))

opt, opt_ind, _ = compute_optimum_pricing(prices=prices, bid=bid, user_classes=classes)
opt = 19650.958342283884
print(opt, opt_ind)

# Number of experiments
noe = 50
ts_final_rewards = []
split_record = []
for exp in tqdm(range(noe)):
    first_learner = TS_Learner(n_arms=n_arms, n_features=2, names=names)
    missing_days = T
    env = Environment(n_arms, rates, margins, poissons, return_time=return_time, prices=prices, bid=bid)
    return_ts, return_ts_classes, return_ts_dequy, current_rewards, dd, split, arms, _ = context_split(days=missing_days,
                                                                                                    current_learner=
                                                                                                    first_learner,
                                                                                                    classes=classes,
                                                                                                    env=env)

    return_ts_l, return_ts_classes_l, return_ts_dequy_l, current_rewards_l, dd_l, split_l, arms_l = [], \
                                                                                                    [], [], [], [], [], []

    # FIRST SPLIT

    for i in range(len(return_ts)):
        if dd != 0:
            a, b, c, d, e, f, g, _ = context_split(days=dd, current_learner=return_ts[i], classes=return_ts_classes[i],
                                                env=env, ts_dequy=return_ts_dequy[i])

            return_ts_l.append(a)
            return_ts_classes_l.append(b)
            return_ts_dequy_l.append(c)
            current_rewards_l.append(d)
            dd_l.append(e)
            split_l.append(f)
            arms_l.append(g)

    # SECOND SPLIT
    sec_return_ts_l, sec_return_ts_classes_l, sec_return_ts_dequy_l, sec_current_rewards_l, sec_dd_l, sec_split_l, sec_arms_l = [], \
                                                                                                                                [], [], [], [], [], []
    for lea in range(len(return_ts_l)):
        for i in range(len(return_ts_l[lea])):
            if dd_l != [] and dd_l[lea] != 0:
                a, b, c, d, e, f, g, _ = context_split(days=dd_l[lea], current_learner=return_ts_l[lea][i],
                                                    classes=return_ts_classes_l[lea][i],
                                                    env=env, ts_dequy=return_ts_dequy_l[lea][i])

                sec_return_ts_l.append(a)
                sec_return_ts_classes_l.append(b)
                sec_return_ts_dequy_l.append(c)
                sec_current_rewards_l.append(d)
                sec_dd_l.append(e)
                sec_split_l.append(f)
                sec_arms_l.append(g)

    rew = aggregate_rewards(current_rewards, current_rewards_l, sec_current_rewards_l, missing_days)
    ts_final_rewards.append(rew)

    split_this = []
    for first in return_ts_classes_l:
        if not isinstance(first[0], list):
            split_this.append(first)
        else:
            for second in first:
                split_this.append(second)
    split_record.append({'numb': len(split_this), 'splits': split_this})


plot_reward(opt=opt, final_rewards=[ts_final_rewards], names=['TS_Learner'], color_list=['r'])
plot_regret(opt=opt, final_rewards=[np.array(ts_final_rewards)], names=['TS_Learner'], color_list=['r'])
