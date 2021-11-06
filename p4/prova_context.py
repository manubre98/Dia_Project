from tqdm import tqdm

from p3.p3_Environment import *
from p4.p4_functions import *
from utils import *

np.random.seed(4)

# Environment
n_arms = 10
prices = np.linspace(7.5, 11.5, n_arms)
margins = margin(prices)
T = 1000
return_time = 30
table = None

# People
names = [['Young ', 'Old'], ['Not Sporty', 'Sporty']]
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
opt_vec = []
for i in range(np.size(prices)):
    opt_arr = [r[i] for r in rates]
    opt_vec.append(margins[i] * np.sum((np.array(poissons) + 1) * np.array(accesses) * np.array(opt_arr)))

opt_ind = np.argmax(np.array(opt_vec))
opt = np.max(np.array(opt_vec))
print(opt, opt_ind)


# Number of experiments
noe = 100
ts_final_rewards = []
split_history = []
for e in tqdm(range(noe), desc='Number of experiments'):

    # Create Environment
    env = Environment(n_arms, rates, margins, poissons, return_time=return_time)

    # Create Learner
    ts_learner = TS_Learner(n_arms, names=names)
    ts_rewards = []

    # generate empty deque
    ts_dequy = deque({'arm': 0, 'average_returns': 0, 'sample': 0} for _ in range(return_time))

    arms = []
    # Simulate experiments
    for d in range(T):

        # choose arm
        ts_pulled_arm = ts_learner.pull_arm(margins)
        arms.append(ts_pulled_arm)

        # simulate accesses
        # empty daily reward
        ts_daily = {'reward': 0, 'successes': 0, 'returns': 0}
        clicks = []
        purchases = []
        returns = []

        for c in range(len(classes)):
            n_trials = int(accesses[c])
            # quanti acquisti
            ts_successes, ts_reward, ts_class_returns = env.round(ts_pulled_arm, c, n_trials)
            # aggiorno beta
            ts_learner.update(ts_pulled_arm, ts_successes, n_trials)

            # aggiorno reward_giornaliera
            ts_daily['reward'] += ts_reward
            ts_daily['successes'] += ts_successes
            ts_daily['returns'] += ts_class_returns

            clicks.append(n_trials)
            purchases.append(ts_successes)
            returns.append(ts_class_returns)

        r_age = Register(pulled_arm=ts_pulled_arm, clicks0=clicks[0] + clicks[2],
                         purchases0=purchases[0] + purchases[2],
                         clicks1=clicks[1] + clicks[3], purchases1=purchases[1] + purchases[3],
                         returns0=returns[0] + returns[2], returns1=returns[1] + returns[3])

        r_sport = Register(pulled_arm=ts_pulled_arm, clicks0=clicks[0] + clicks[1],
                           purchases0=purchases[0] + purchases[1],
                           clicks1=clicks[2] + clicks[3], purchases1=purchases[2] + purchases[3],
                           returns0=returns[0] + returns[1], returns1=returns[2] + returns[3])

        ts_learner.table.update([r_age, r_sport])

        # work on dequy
        ts_dicty = {'arm': ts_pulled_arm, 'average_returns': ts_daily['returns'] / ts_daily['successes'],
                    'sample': ts_daily['successes']}
        ts_dequy.append(ts_dicty)
        ts_learner.update_poisson(ts_dequy.popleft())

        # save daily reward
        ts_rewards.append(ts_daily['reward'])

        if d == 990:
            split_history.append(ts_learner.evaluate_means(margins, chosen_arms=arms))

    ts0 = TS_Learner(n_arms=n_arms)
    ts1 = TS_Learner(n_arms=n_arms)

    ts0.inherit(ts_learner, split_history[e], 0)
    ts1.inherit(ts_learner, split_history[e], 1)
    # ts_rewards.insert(0, 0)
    ts_final_rewards.append(ts_rewards)

# PLOT SECTION
plot_regret(opt, [ts_final_rewards], names=['TS_Learner'], color_list=['r'])
print("done")
