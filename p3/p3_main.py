from collections import deque

from tqdm import tqdm

from p3.p3_Environment import *
from p3.p3_TS_Learner import *
from p3.p3_UCB1 import *
from utils import *
from p4.p4_functions import *

# Environment
n_arms = 10
prices = np.linspace(5, 14, n_arms)
margins = margin(prices)
T = 365
return_time = 30

# People
bid = 1.2
accesses = []
poissons = []
rates = []
for c in classes:
    accesses.append(np.round(c.clicks(bid)))
    poissons.append(c.poisson)
    rates.append(c.conversion_rate(prices))

# print(rates)
# Find optimal index
opt, opt_ind, opt_vec = compute_optimum_pricing(prices=prices, bid=bid, user_classes=classes)
print(opt, opt_ind)
print(opt_vec)
optbest = 19689.956148631187


# Number of experiments
noe = 50
ts_final_rewards = []
ucb_final_rewards = []

for e in tqdm(range(noe), desc='Number of experiments'):

    # Create Environment
    env = Environment(n_arms, rates, margins, poissons, return_time=return_time, prices=prices, bid=bid)
    for c in classes:
        c.evaluate(env.bid, env.prices)
    # Create Learner
    ts_learner = TS_Learner(n_arms)
    ts_rewards = []

    ucb_learner = UCB1(n_arms)
    ucb_rewards = []

    # generate empty deque
    ts_dequy = deque({'arm': 0, 'average_returns': 0, 'sample': 0} for _ in range(return_time))
    ucb_dequy = deque({'arm': 0, 'average_returns': 0, 'sample': 0} for _ in range(return_time))

    # Simulate experiments
    for d in range(T):

        # choose arm
        ts_pulled_arm = ts_learner.pull_arm(margins)
        ucb_pulled_arm = ucb_learner.pull_arm(margins)

        # simulate accesses
        # empty daily reward
        ts_daily = {'reward': 0, 'successes': 0, 'returns': 0}
        ucb_daily = {'reward': 0, 'successes': 0, 'returns': 0, 'trials': 0}

        for i, c in enumerate(classes):
            n_trials = int(c.accesses)
            # quanti acquisti
            ts_successes, ts_reward, ts_class_returns = env.round(ts_pulled_arm, c, n_trials)
            ucb_successes, ucb_reward, ucb_class_returns = env.round(ucb_pulled_arm, c, n_trials)
            # aggiorno beta
            ts_learner.update(ts_pulled_arm, ts_successes, n_trials)

            # aggiorno reward_giornaliera
            ts_daily['reward'] += ts_reward
            ts_daily['successes'] += ts_successes

            ucb_daily['successes'] += ucb_successes
            ucb_daily['reward'] += ucb_reward
            ucb_daily['trials'] += n_trials

            ts_daily['returns'] += ts_class_returns
            ucb_daily['returns'] += ucb_class_returns

        ucb_learner.update(ucb_pulled_arm, ucb_daily['successes'], ucb_daily['trials'])

        # work on dequy
        ts_dicty = {'arm': ts_pulled_arm, 'average_returns': ts_daily['returns'] / ts_daily['successes'],
                    'sample': ts_daily['successes']}
        ts_dequy.append(ts_dicty)
        ts_learner.update_poisson(ts_dequy.popleft())

        ucb_dicty = {'arm': ucb_pulled_arm, 'average_returns': ucb_daily['returns'] / ucb_daily['successes'],
                     'sample': ucb_daily['successes']}
        ucb_dequy.append(ucb_dicty)
        ucb_learner.update_poisson(ucb_dequy.popleft())

        # save daily reward
        ts_rewards.append(ts_daily['reward'])
        ucb_rewards.append(ucb_daily['reward'])

    # ts_rewards.insert(0, 0)
    # ucb_rewards.insert(0, 0)
    ts_final_rewards.append(ts_rewards)
    ucb_final_rewards.append(ucb_rewards)


plot_regret(opt, [ts_final_rewards], names=['TS_Learner'], color_list=['r'])
plot_regret(opt, [ucb_final_rewards], names=['UCB1'], color_list=['g'])
plot_regret2(opt, [ts_final_rewards, ucb_final_rewards], names=['TS_Learner', 'UCB1'], color_list=['r', 'g'])

plot_reward(opt, [ts_final_rewards], names=['TS_Learner'], color_list=['r'], optbest=optbest)
plot_reward(opt, [ucb_final_rewards], names=['UCB1'], color_list=['g'], optbest=optbest)
plot_reward2(opt, [ts_final_rewards, ucb_final_rewards], names=['TS_Learner', 'UCB1'], color_list=['r', 'g'], optbest=optbest)
print("done")
