import warnings

from tqdm import tqdm

from p5.GPTS_Learner import *
from p5.p5_BiddingEnvironment import *
from p5.p5_functions import *
from utils import *

np.random.seed(130503) ##56

warnings.filterwarnings("ignore")

#Environment
n_arms = 10
min_bid = 0.6
max_bid = 1.5
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 200

names = ['Young ', 'Old', 'Not Sporty', 'Sporty']

price = 9
T = 365
n_experiments = 50

gpts_rewards_per_experiment = []

opt, opt_ind, opt_vec = compute_optimum_bidding(bids, price, classes)
print(opt, opt_ind, bids[opt_ind])
print(opt_vec)

for e in tqdm(range(0, n_experiments)):
    env = BiddingEnvironment(bids=bids, sigma = sigma, user_classes=classes, price=price, n_arms=n_arms)
    gpts_learner = GPTS_Learner(n_arms, arms=bids, names=names, price=price)

    for t in range(T):
        #GP Thompson Sampling
        pulled_arm = gpts_learner.pull_arm()
        reward = 0
        for i, c in enumerate(classes):
            reward += env.round(pulled_arm, i)

        gpts_learner.update(pulled_arm, reward)

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)

plt.figure(0)
plot_reward(opt, [gpts_rewards_per_experiment], names=['TS_Learner'], color_list=['r'])
plt.figure(1)
plot_regret(opt, [gpts_rewards_per_experiment], names=['TS_Learner'], color_list=['r'])

plt.figure(2)
plt.title("REGRET - {} simulations".format(len(gpts_rewards_per_experiment)))
plt.xlabel("t")
plt.ylabel("Regret")
leg = "GPTS Learner"
maxes = []
for line in [gpts_rewards_per_experiment]:
    curve = np.nancumsum(np.nanmean(opt - line, axis=0))
    maxes.append(np.max(curve))
    print(line)
    plt.plot(curve, "r")
    for i in range(len(gpts_rewards_per_experiment)):
        plt.plot(np.nancumsum(opt - line[i], axis=0), "g", alpha=1 / np.power(len(gpts_rewards_per_experiment), 2/3))

plt.ylim([0, 1.8 * np.max(maxes, axis=0)])
#plt.legend(leg)
plt.show()

plt.figure(3)
plt.title("REWARD - {} simulations".format(len(gpts_rewards_per_experiment)))
plt.xlabel("t")
plt.ylabel("Reward")
leg = "GPTS Learner"
maxes = []
opt_line = np.ones(shape=len(gpts_rewards_per_experiment)) * opt
plt.plot(opt_line, 'k')
for line in [gpts_rewards_per_experiment]:
    curve = np.nanmean(line, axis=0)
    maxes.append(np.max(curve))
    print(line)
    plt.plot(curve, "r")
    for i in range(len(gpts_rewards_per_experiment)):
        plt.plot(line[i], "g", alpha=1 / np.power(len(gpts_rewards_per_experiment), 2/3))
plt.ylim([15000, 1.2 * np.max(maxes, axis=0)])
#plt.legend(leg)
plt.show()