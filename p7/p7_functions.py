from p3.p3_Environment import *
from p4.p4_functions import *
from p5.p5_functions import *
from utils import *
from p5.GPTS_Learner import *
from p5.p5_BiddingEnvironment import *
from tqdm import tqdm
import warnings


def pricing_context(prices=np.linspace(7.5, 11.5, 10), n_arms=10, T=365, classes=classes, bid=1.2, actual_price=0,
                    reward_vec=None):
    # Environment
    margins = margin(prices)
    return_time = 30
    table = None
    starting_point = len(reward_vec) - T
    # People
    names = [c.name for c in classes]
    accesses = []
    poissons = []
    rates = []
    for c in classes:
        accesses.append(np.round(c.clicks(bid)))
        poissons.append(c.poisson)
        rates.append(c.conversion_rate(prices))

    opt, opt_ind, _ = compute_optimum_pricing(prices=prices, bid=bid, user_classes=classes)
    print(opt, opt_ind)

    # Number of experiments
    first_learner = TS_Learner(n_arms=n_arms, n_features=2, names=names)
    missing_days = 365
    env = Environment(n_arms, rates, margins, poissons, return_time=return_time, prices=prices, bid=bid)
    return_ts, return_ts_classes, return_ts_dequy, current_rewards, dd, split, arms, best_price = context_split(
        days=missing_days,
        current_learner=
        first_learner,
        classes=classes,
        env=env)

    reward_vec[starting_point:(starting_point + len(current_rewards))] += current_rewards
    starting_point = starting_point + len(current_rewards)

    return_ts_l, return_ts_classes_l, return_ts_dequy_l, current_rewards_l, dd_l, split_l, arms_l, best_price_l = [], \
                                                                                                                  [], [], [], [], [], [], []

    # FIRST SPLIT

    for i in range(len(return_ts)):
        if dd != 0:
            a, b, c, d, e, f, g, h = context_split(days=dd, current_learner=return_ts[i], classes=return_ts_classes[i],
                                                   env=env, ts_dequy=return_ts_dequy[i])

            return_ts_l.append(a)
            return_ts_classes_l.append(b)
            return_ts_dequy_l.append(c)
            current_rewards_l.append(d)
            dd_l.append(e)
            split_l.append(f)
            arms_l.append(g)
            best_price_l.append(h)

    best_price_return = double_nested_loop(best_price_l)

    dd_return = dd_creation(dd_l)
    if current_rewards_l != []:
        for r in current_rewards_l:
            reward_vec[starting_point:(starting_point + len(r))] += r

    split_this = []
    for first in return_ts_classes_l:
        if not isinstance(first[0], list):
            split_this.append(first)
        else:
            for second in first:
                split_this.append(second)

    return split_this, dd_return, best_price_return


def dd_creation(dd_l):
    dd_return = None
    if dd_l == []:
        dd_return = [0]
    if dd_l[0] == 0 and dd_l[1] == 0:
        dd_return = [0, 0]
    if dd_l[0] != 0 and dd_l[1] == 0:
        dd_return = [dd_l[0], dd_l[0], 0]
    if dd_l[0] == 0 and dd_l[1] != 0:
        dd_return = [0, dd_l[1], dd_l[1]]
    if dd_l[0] != 0 and dd_l[1] != 0:
        dd_return = [dd_l[0], dd_l[0], dd_l[1], dd_l[1]]

    return dd_return


def bidding_context(n_arms=10, bids=None, price=9, T=365, sigma=300, actual_bid=0, reward_vec=None, classes=classes):
    warnings.filterwarnings("ignore")
    names = [c.name for c in classes]

    opt, opt_ind, opt_vec = compute_optimum_bidding(bids, price, classes)
    # print(opt, opt_ind, bids[opt_ind])
    # print(opt_vec)
    starting_point = len(reward_vec) - T
    env = BiddingEnvironment(bids=bids, sigma=sigma, user_classes=classes, price=price, n_arms=n_arms)
    gpts_learner = GPTS_Learner(n_arms, arms=bids, names=names, price=price)
    arms = []

    t = 0
    for t in range(T):
        # GP Thompson Sampling
        pulled_arm = gpts_learner.pull_arm()
        reward = 0
        for index_c, c in enumerate(classes):
            reward += env.round(pulled_arm, index_c)

        gpts_learner.update(pulled_arm, reward)
        arms.append(pulled_arm)
        reward_vec[starting_point + t] += reward
        if check_convergence(arms, crit=0.8)[0]:
            best_bid = bids[check_convergence(arms, crit=0.8)[1]]
            return best_bid, T - 1 - t

    return actual_bid, T - 1 - t


def check_convergence_p7(arms, thres=400, crit=0.7):
    conv = False
    unique_elements, count_elements = np.unique(np.array(arms[-int(0.10 * thres):]), return_counts=True)
    if len(arms) >= 0.10 * thres and np.max(count_elements) > crit * len(arms[-int(0.10 * thres):]):
        conv = True
    best_arm = np.bincount(arms[-int(0.10 * thres):]).argmax()
    return conv, best_arm


def compute_optimum_p7(bids, prices, user_classes):
    n = len(bids)
    mat = np.zeros(shape=(n, n))

    for bi in range(len(bids)):
        for j in range(len(prices)):
            mat[bi, j] = obj_fun(user_classes, bids[bi], prices[j])

    imax, jmax = np.unravel_index(np.argmax(mat), (n, n))

    return bids[imax], prices[jmax], np.max(mat), mat


def update_splitting_table(pricing_learner, pricing_pulled_arm, clicks, purchases, returns):

    if len(clicks) == 4:
        r0 = Register(pulled_arm=pricing_pulled_arm, _class=0, clicks=clicks[0],
                      purchases=purchases[0], returns=returns[0])
        r1 = Register(pulled_arm=pricing_pulled_arm, _class=1, clicks=clicks[1],
                      purchases=purchases[1], returns=returns[1])
        r2 = Register(pulled_arm=pricing_pulled_arm, _class=2, clicks=clicks[2],
                      purchases=purchases[2], returns=returns[2])
        r3 = Register(pulled_arm=pricing_pulled_arm, _class=3, clicks=clicks[3],
                      purchases=purchases[3], returns=returns[3])

        pricing_learner.table.update([r0, r1, r2, r3])

    elif len(clicks) == 2:
        r0 = Register(pulled_arm=pricing_pulled_arm, _class=0, clicks=clicks[0],
                      purchases=purchases[0], returns=returns[0])
        r1 = Register(pulled_arm=pricing_pulled_arm, _class=1, clicks=clicks[1],
                      purchases=purchases[1], returns=returns[1])
        pricing_learner.table.update([r0, r1])

    return
