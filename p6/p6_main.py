import warnings

from tqdm import tqdm

from p4.p4_functions import *
from p6.p6_BiddingEnvironment import *
from p6.p6_GPTS_Learner import *
from p6.p6_PricingEnvironment import *

warnings.filterwarnings("ignore")

np.random.seed(542654)

# Environment
n_arms = 10
prices = np.linspace(9, 9, n_arms)
bids = np.linspace(0.6, 1.5, n_arms)
margins = margin(prices)
T = 365
return_time = 30
bidding_pulled_arm = 1.2

# print(rates)
# Find optimal index
opt, opt_ind, opt_vec = compute_optimum_pricing(prices=prices, bid=bidding_pulled_arm, user_classes=classes)
print(opt, opt_ind)
print(opt_vec)

poissons = []
rates = []
for c in classes:
    poissons.append(c.poisson)
    rates.append(c.conversion_rate(prices))

# Number of experiments
noe = 50
ts_final_rewards = []
ucb_final_rewards = []
conv_arms = []
acc_sigma = 50
cost_sigma = 0.1

# Create Environment
pr_env = PricingEnvironment6(n_arms, rates, margins, poissons, return_time=return_time, prices=prices)
bid_env = BiddingEnvironment6(bids, acc_sigma, cost_sigma, classes, n_arms)

final_bids = []
for e in tqdm(range(noe), desc='Number of experiments'):

    # for c in classes:
    #    c.evaluate(env.bidding_pulled_arm, env.prices)

    # Create Learner
    pricing_learner = TS_Learner(n_arms)
    bidding_learner = GPTS_Learner6(n_arms, arms=bids)
    ts_rewards = []

    # generate empty deque
    ts_dequy = deque({'arm': 0, 'average_returns': 0, 'sample': 0} for _ in range(return_time))
    arms = []
    bids_p = []
    first = True

    # Simulate experiments
    for d in range(T):

        # choose arms
        pricing_pulled_arm = pricing_learner.pull_arm(margins)
        arms.append(pricing_pulled_arm)
        bidding_pulled_arm = bidding_learner.pull_arm(pricing_learner=pricing_learner, price_idx=pricing_pulled_arm,
                                                      margins=margins)

        # simulate accesses
        # empty daily reward
        ts_daily = {'reward': 0, 'successes': 0, 'returns': 0, 'clicks': 0}
        costpc = 0
        for i, c in enumerate(classes):
            n_trials, costpc = bid_env.round(pulled_arm=bidding_pulled_arm, user_c=i)
            # quanti acquisti
            ts_successes, ts_class_returns = pr_env.round(pricing_pulled_arm, c, n_trials)
            # aggiorno beta
            pricing_learner.update(pricing_pulled_arm, ts_successes, n_trials)

            # aggiorno reward_giornaliera
            ts_daily['reward'] += (ts_successes + ts_class_returns) * margin(prices[pricing_pulled_arm]) - n_trials * costpc
            ts_daily['successes'] += ts_successes
            ts_daily['returns'] += ts_class_returns
            ts_daily['clicks'] += n_trials

        # work on dequy
        try:
            av_ret = ts_daily['returns'] / ts_daily['successes']
        except ZeroDivisionError:
            av_ret = 0

        ts_dicty = {'arm': pricing_pulled_arm, 'average_returns': av_ret,
                    'sample': ts_daily['successes']}
        ts_dequy.append(ts_dicty)
        pricing_learner.update_poisson(ts_dequy.popleft())
        bidding_learner.update(pulled_arm=bidding_pulled_arm, costs=costpc, clicks=ts_daily['clicks'])
        # save daily reward
        ts_rewards.append(ts_daily['reward'])
        if check_convergence(arms)[0] and first:
            Didi = {'arm': check_convergence(arms)[1], 'day': d, 'exp': e}
            first = False
            conv_arms.append(Didi)
            # print(check_convergence(arms)[1])
        bids_p.append(bidding_pulled_arm)
        if check_convergence(bids_p, crit=0.7)[0]:
            print('Conv bid', check_convergence(bids_p, crit=0.7)[1])

    final_bids.append(bids_p)
    # print(arms)
    ts_rewards.insert(0, 0)
    ts_final_rewards.append(ts_rewards)
final_bids = np.array(final_bids)
# plot_regret(opt, [ts_final_rewards], names=['TS_Learner'], color_list=['r'])
print("done")
print(len(conv_arms))
print(np.mean([dic['day'] for dic in conv_arms]))
print(np.unique([dic['arm'] for dic in conv_arms], return_counts=True))

vettore = np.zeros(shape=n_arms)
for i, b in enumerate(bids):
    for c in classes:
        vettore[i] += c.clicks(b)

plt.figure(0)
plt.title("REGRET - {} simulations".format(len(ts_final_rewards)))
plt.xlabel("t")
plt.ylabel("Regret")
#leg = "GPTS Learner"
maxes = []
for line in [ts_final_rewards]:
    curve = np.nancumsum(np.nanmean(opt - line, axis=0))
    maxes.append(np.max(curve))
    print(line)
    plt.plot(curve, "r")
    for i in range(len(ts_final_rewards)):
        plt.plot(np.nancumsum(opt - line[i], axis=0), "g", alpha=1 / np.power(len(ts_final_rewards), 2/3))

plt.ylim([0, 1.8 * np.max(maxes, axis=0)])
#plt.legend(leg)
plt.show()

plt.figure(1)
plt.title("REWARD - {} simulations".format(len(ts_final_rewards)))
plt.xlabel("t")
plt.ylabel("Reward")
#leg = "GPTS Learner"
maxes = []
opt_line = np.ones(shape=len(ts_final_rewards)) * opt
plt.plot(opt_line, 'k')
for line in [ts_final_rewards]:
    curve = np.nanmean(line, axis=0)
    maxes.append(np.max(curve))
    print(line)
    plt.plot(curve, "r")
    for i in range(len(ts_final_rewards)):
        plt.plot(line[i], "g", alpha=1 / np.power(len(ts_final_rewards), 2/3))
plt.ylim([500, 1.4 * np.max(maxes, axis=0)])
#plt.legend(leg)
plt.show()