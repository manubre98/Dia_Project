import matplotlib.pyplot as plt

from p3_Environment import *
from p3_TS_Learner import *

# Environment
n_arms = 30
prices = np.linspace(7, 11, n_arms)
margins = margin(prices)
T = 1000

# People
bid = 1.2
accesses = []
poissons = []
rates = []
for c in classes:
    accesses.append(np.round(c.clicks(bid)))
    poissons.append(c.poisson)
    rates.append(c.conversion_rate(prices))

# Find optimal index
opt_vec = []
print(np.size(prices), len(rates[1]), len(margins))
for i in range(np.size(prices)):
    opt_arr = [r[i] for r in rates]
    opt_vec.append(margins[i] * np.sum(accesses * np.array(opt_arr)))

opt_ind = np.argmax(np.array(opt_vec))
opt = np.max(np.array(opt_vec))
print(opt, opt_ind)

# Create Environment
env = Environment(n_arms, rates, margins)

# Create Learner
ts_learner = TS_Learner(n_arms)
ts_rewards = []
arms = []
# Simulate experiments
for d in range(T):

    # choose arm
    ts_pulled_arm = ts_learner.pull_arm(margins)  # AGGIUNGI POISSON COSì MANUEL SOFFRE
    print(prices[ts_pulled_arm])
    arms.append(ts_pulled_arm)

    # simulate access
    # empty reward_giornaliera
    ts_daily_reward = 0

    for c in range(3):
        for i in range(int(accesses[c])):
            # acquista o no
            success, reward = env.round(ts_pulled_arm, c)
            # aggiorno beta
            ts_learner.update(ts_pulled_arm, success, reward)
            # aggiorno reward_giornaliera
            ts_daily_reward += reward

    # salvo ts_reward
    ts_rewards.append(ts_daily_reward)

    # choose arm
    # simulate access
    # empty ts_reward

    # for num in accesses:
    #    for i in range(num):
    # acquista o no
    # aggiorno beta
    # aggiorno ts_reward

    # salvo ts_reward
    print(d)

# print(ts_rewards)
# print(opt)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("regret")
plt.plot(np.nancumsum(opt - np.array(ts_rewards), axis=0), 'r')
plt.legend(["TS"])
plt.show()
print("done")

print(ts_learner.beta_parameters)

unique_elements, counts_elements = np.unique(np.array(arms), return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((prices[unique_elements], counts_elements)))

plt.hist(prices[arms])
plt.show()
