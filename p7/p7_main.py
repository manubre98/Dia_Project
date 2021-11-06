from p6.p6_BiddingEnvironment import *
from p6.p6_GPTS_Learner import *
from p6.p6_PricingEnvironment import *
from p7.p7_TS_Learner import *
from p7.p7_functions import *

warnings.filterwarnings("ignore")

np.random.seed(3456789)
exp_params = {'n_arms': 10, 'noe': 5, 'T': 365, 'return_time': 30, 'first2': [False, False, False, False]}
bidding_params = {'min_bid': 0.6, 'max_bid': 1.5, 'acc_sigma': 10, 'cost_sigma': 0.01}

prices = np.linspace(5, 14, exp_params['n_arms'])
bids = np.linspace(bidding_params['min_bid'], bidding_params['max_bid'], exp_params['n_arms'])
margins = margin(prices)
bidding_pulled_arm = 1.2

# Find optimal index
opt_bid, opt_price, optimum, reward_matrix = compute_optimum_p7(prices=prices, bids=bids, user_classes=classes)
print(opt_bid, opt_price, optimum)
opt = 19650.958342283884

poissons = []
rates = []
for c in classes:
    poissons.append(c.poisson)
    rates.append(c.conversion_rate(prices))

# Number of experiments
ts_final_rewards = []
conv_arms = []

# Create Environment
pr_env = PricingEnvironment6(exp_params['n_arms'], rates, margins, poissons, return_time=exp_params['return_time'],
                             prices=prices)
bid_env = BiddingEnvironment6(bids, bidding_params['acc_sigma'], bidding_params['cost_sigma'], classes,
                              exp_params['n_arms'])

final_bids = []
lengths = []
noc = []
for e in tqdm(range(exp_params['noe']), desc='Number of experiments'):
    first = True
    LEARNERS_LIST = []
    starting_point = 0
    split = None

    # Create Learner
    pricing_learner = TS_Learner7(exp_params['n_arms'])
    LEARNERS_LIST.append(pricing_learner)
    bidding_learner = GPTS_Learner6(exp_params['n_arms'], arms=bids)
    # ts_rewards = []

    # generate empty deque
    ts_dequy = deque({'arm': 0, 'average_returns': np.zeros(shape=len(classes)),
                      'sample': np.zeros(shape=len(classes))} for _ in range(pr_env.return_time))
    arms = []
    bids_p = []

    # PRIMO CICLO
    print('\nFIRST CYCLE')
    rewards_c1 = []
    for d in range(starting_point, exp_params['T']):
        starting_point = d+1

        # choose arms
        pricing_pulled_arm = pricing_learner.pull_arm(margins)
        arms.append(pricing_pulled_arm)
        bidding_pulled_arm = bidding_learner.pull_arm(pricing_learner=pricing_learner, price_idx=pricing_pulled_arm,
                                                      margins=margins)

        # simulate accesses
        # empty daily reward
        ts_daily = {'reward': 0, 'successes': 0, 'returns': 0, 'clicks': 0, 'list_acc': []}
        costpc = 0
        clicks, purchases, returns = [], [], []
        ts_dicty = {'arm': pricing_pulled_arm, 'average_returns': np.zeros(shape=len(classes)),
                    'sample': np.zeros(shape=len(classes))}
        for i, c in enumerate(classes):
            n_trials, costpc = bid_env.round(pulled_arm=bidding_pulled_arm, user_c=i)
            c.accesses = n_trials
            # quanti acquisti
            ts_successes, ts_class_returns = pr_env.round(pricing_pulled_arm, c, n_trials)
            # aggiorno beta
            pricing_learner.update(pricing_pulled_arm, ts_successes, n_trials)

            # aggiorno reward_giornaliera
            try:
                avi = ts_class_returns / ts_successes
            except ZeroDivisionError:
                avi = 0
            ts_dicty['average_returns'][i] = avi
            ts_dicty['sample'][i] = ts_successes
            ts_daily['reward'] += (ts_successes + ts_class_returns) * margin(
                prices[pricing_pulled_arm]) - n_trials * costpc
            ts_daily['successes'] += ts_successes
            ts_daily['returns'] += ts_class_returns
            ts_daily['clicks'] += n_trials
            ts_daily['list_acc'].append(n_trials)
            clicks.append(n_trials)
            purchases.append(ts_successes)
            returns.append(ts_class_returns)

        rewards_c1.append(ts_daily['reward'])
        bidding_learner.update(pulled_arm=bidding_pulled_arm, costs=costpc, clicks=ts_daily['clicks'])

        # ts_rewards.append(ts_daily['reward'])
        update_splitting_table(pricing_learner=pricing_learner, returns=returns, clicks=clicks, purchases=purchases,
                               pricing_pulled_arm=pricing_pulled_arm)

        # work on dequy
        ts_dequy.append(ts_dicty)
        pricing_learner.update_poisson_context(ts_dequy.popleft())

        bids_p.append(bidding_pulled_arm)
        if check_convergence(bids_p, crit=0.7)[0] and first:
            first = False
            print('\nConv bid', check_convergence(bids_p, crit=0.7)[1], 'Convergence day', d)

        if check_convergence(arms)[0]:
            Didi = {'arm': check_convergence(arms)[1], 'day': d, 'exp': e}
            conv_arms.append(Didi)
            print('\nConvergence price', check_convergence(arms)[1], '-', 'Convergence Day', d)
            split, proposed_prices = pricing_learner.evaluate_means(margins=margins, chosen_arms=arms,
                                                                    list_acc=ts_daily['list_acc'])
            print('Proposed Split: ', split, 'Prices: ', proposed_prices)

        if pricing_learner.table.n_classes == 4 and split != 2 and split is not None:
            class0, class1 = None, None
            ts0_dequy, ts1_dequy = None, None
            if split == 0:
                class0, class1 = [classes[0], classes[2]], [classes[1], classes[3]]
                ts0_dequy, ts1_dequy = adapt_dequy(ts_dequy, [0, 2]), adapt_dequy(ts_dequy, [1, 3])
            if split == 1:
                class0, class1 = [classes[0], classes[1]], [classes[2], classes[3]]
                ts0_dequy, ts1_dequy = adapt_dequy(ts_dequy, [0, 1]), adapt_dequy(ts_dequy, [2, 3])
            ts0 = TS_Learner7(n_arms=pr_env.n_arms, names=[class0[0].name, class0[1].name], n_features=1)
            ts0.inherit(pricing_learner, split, 0)
            ts1 = TS_Learner7(n_arms=pr_env.n_arms, names=[class1[0].name, class1[1].name], n_features=1)
            ts1.inherit(pricing_learner, split, 1)
            LEARNERS_LIST = [ts0, ts1]
            return_ts_classes = [class0, class1]
            return_ts_dequy = [ts0_dequy, ts1_dequy]
            return_bid_environment = [BiddingEnvironment6(bids, acc_sigma=bidding_params['acc_sigma'],
                                                          cost_sigma=bidding_params['cost_sigma'], user_classes=class0,
                                                          n_arms=exp_params['n_arms']),
                                      BiddingEnvironment6(bids, acc_sigma=bidding_params['acc_sigma'],
                                                          cost_sigma=bidding_params['cost_sigma'], user_classes=class1,
                                                          n_arms=exp_params['n_arms']),
                                      ]
            return_arms = [[], []]
        if len(LEARNERS_LIST) != 1:
            break
    if starting_point == (exp_params['T'] - 1): starting_point += 1
    split = None

    # SECONDO CICLO
    print('SECOND CYCLE')
    rewards_c2 = []
    for d in range(starting_point, exp_params['T']):
        starting_point = d+1

        pricing_pulled_arms = []
        for index, pricing_learner in enumerate(LEARNERS_LIST):
            arm = pricing_learner.pull_arm(margins)
            pricing_pulled_arms.append(arm)
            return_arms[index].append(arm)

        bidding_pulled_arm = bidding_learner.pull_arm_context(learners_list=LEARNERS_LIST,
                                                              price_idx_list=pricing_pulled_arms,
                                                              margins=[margin(prices[p]) for p in pricing_pulled_arms],
                                                              classes_list=return_ts_classes)
        bids_p.append(bidding_pulled_arm)
        access_count = np.zeros(shape=len(LEARNERS_LIST))
        cpc_array = np.zeros(shape=len(LEARNERS_LIST))

        compl_reward = 0
        for index, pricing_learner in enumerate(LEARNERS_LIST):

            ts_daily = {'reward': 0, 'successes': 0, 'returns': 0, 'clicks': 0, 'list_acc': []}
            costpc = 0
            clicks, purchases, returns = [], [], []
            ts_dicty = {'arm': pricing_pulled_arms[index],
                        'average_returns': np.zeros(shape=len(return_ts_classes[index])),
                        'sample': np.zeros(shape=len(return_ts_classes[index]))}

            for i, c in enumerate(return_ts_classes[index]):

                n_trials, costpc = return_bid_environment[index].round(pulled_arm=bidding_pulled_arm, user_c=i)
                c.accesses = n_trials
                cpc_array[index] += costpc * n_trials
                # quanti acquisti
                ts_successes, ts_class_returns = pr_env.round(pricing_pulled_arms[index], c, n_trials)
                # aggiorno beta
                pricing_learner.update(pricing_pulled_arms[index], ts_successes, n_trials)

                # aggiorno reward_giornaliera
                try:
                    avi = ts_class_returns / ts_successes
                except ZeroDivisionError:
                    avi = 0
                ts_dicty['average_returns'][i] = avi
                ts_dicty['sample'][i] = ts_successes
                # ts_daily['reward'] += ts_reward
                compl_reward += margin(prices[pricing_pulled_arms[index]]) * (
                            ts_successes + ts_class_returns) - n_trials * costpc
                ts_daily['successes'] += ts_successes
                ts_daily['returns'] += ts_class_returns
                ts_daily['clicks'] += n_trials
                ts_daily['list_acc'].append(n_trials)
                clicks.append(n_trials)
                purchases.append(ts_successes)
                returns.append(ts_class_returns)

            return_ts_dequy[index].append(ts_dicty)
            access_count[index] += ts_daily['clicks']
            # ts_rewards.append(ts_daily['reward'])
            update_splitting_table(pricing_learner=pricing_learner, returns=returns, clicks=clicks, purchases=purchases,
                                   pricing_pulled_arm=pricing_pulled_arms[index])
            pricing_learner.update_poisson_context(return_ts_dequy[index].popleft())

            if check_convergence(return_arms[index])[0]:
                Didi = {'arm': check_convergence(return_arms[index])[1], 'day': d, 'exp': e}
                # conv_arms.append(Didi)
                print('\nIndex:', index)
                print('Convergence price', check_convergence(return_arms[index])[1], ' - ', 'Convergence Day', d)
                split, proposed_prices = pricing_learner.evaluate_means(margins=margins, chosen_arms=arms,
                                                                        list_acc=ts_daily['list_acc'])
                print(split, proposed_prices)

            if pricing_learner.table.n_classes == 2 and split == 0:
                class0 = return_ts_classes[index][0]
                class1 = return_ts_classes[index][1]
                ts0 = TS_Learner7(n_arms=pr_env.n_arms, names=[class0.name], n_features=0)
                ts0.inherit(pricing_learner, split, 0)
                ts0_dequy = adapt_dequy(return_ts_dequy[index], [0])
                ts1 = TS_Learner7(n_arms=pr_env.n_arms, names=[class1.name], n_features=0)
                ts1.inherit(pricing_learner, split, 1)
                ts1_dequy = adapt_dequy(return_ts_dequy[index], [1])
                LEARNERS_LIST = [ts0, ts1, LEARNERS_LIST[1 - index]]
                return_ts_classes = [[class0], [class1], return_ts_classes[1 - index]]
                return_ts_dequy = [ts0_dequy, ts1_dequy, return_ts_dequy[1 - index]]
                return_bid_environment = [BiddingEnvironment6(bids, acc_sigma=bidding_params['acc_sigma'],
                                                              cost_sigma=bidding_params['cost_sigma'],
                                                              user_classes=[class0],
                                                              n_arms=exp_params['n_arms']),
                                          BiddingEnvironment6(bids, acc_sigma=bidding_params['acc_sigma'],
                                                              cost_sigma=bidding_params['cost_sigma'],
                                                              user_classes=[class1],
                                                              n_arms=exp_params['n_arms']),
                                          return_bid_environment[1 - index]
                                          ]
                return_arms = [[], [], return_arms[1 - index]]
                break

        rewards_c2.append(compl_reward)
        if check_convergence(bids_p, crit=0.7)[0] and first:
            first = False
            print('\nConv bid', check_convergence(bids_p, crit=0.7)[1], 'Convergence Day', d)

        costpc = np.sum(cpc_array) / np.sum(access_count)
        bidding_learner.update(pulled_arm=bidding_pulled_arm, costs=costpc, clicks=np.sum(access_count))
        if len(LEARNERS_LIST) != 2:
            break

    split = None
    if starting_point == (exp_params['T'] - 1): starting_point += 1

    # TERZO CICLO
    print('THIRD CYCLE')
    rewards_c3 = []
    for d in range(starting_point, exp_params['T']):
        starting_point = d+1

        pricing_pulled_arms = []
        compl_reward = 0
        for index, pricing_learner in enumerate(LEARNERS_LIST):
            arm = pricing_learner.pull_arm(margins)
            pricing_pulled_arms.append(arm)
            return_arms[index].append(arm)

        bidding_pulled_arm = bidding_learner.pull_arm_context(learners_list=LEARNERS_LIST,
                                                              price_idx_list=pricing_pulled_arms,
                                                              margins=[margin(prices[p]) for p in pricing_pulled_arms],
                                                              classes_list=return_ts_classes)
        bids_p.append(bidding_pulled_arm)
        access_count = np.zeros(shape=len(LEARNERS_LIST))
        cpc_array = np.zeros(shape=len(LEARNERS_LIST))
        for index, pricing_learner in enumerate(LEARNERS_LIST):

            ts_daily = {'reward': 0, 'successes': 0, 'returns': 0, 'clicks': 0, 'list_acc': []}
            costpc = 0
            clicks, purchases, returns = [], [], []
            ts_dicty = {'arm': pricing_pulled_arms[index],
                        'average_returns': np.zeros(shape=len(return_ts_classes[index])),
                        'sample': np.zeros(shape=len(return_ts_classes[index]))}

            for i, c in enumerate(return_ts_classes[index]):

                n_trials, costpc = return_bid_environment[index].round(pulled_arm=bidding_pulled_arm, user_c=i)
                c.accesses = n_trials
                cpc_array[index] += costpc * n_trials
                # quanti acquisti
                ts_successes, ts_class_returns = pr_env.round(pricing_pulled_arms[index], c, n_trials)
                # aggiorno beta
                pricing_learner.update(pricing_pulled_arms[index], ts_successes, n_trials)

                # aggiorno reward_giornaliera
                try:
                    avi = ts_class_returns / ts_successes
                except ZeroDivisionError:
                    avi = 0
                ts_dicty['average_returns'][i] = avi
                ts_dicty['sample'][i] = ts_successes
                # ts_daily['reward'] += ts_reward
                compl_reward += margin(prices[pricing_pulled_arms[index]]) * (
                            ts_successes + ts_class_returns) - n_trials * costpc
                ts_daily['successes'] += ts_successes
                ts_daily['returns'] += ts_class_returns
                ts_daily['clicks'] += n_trials
                ts_daily['list_acc'].append(n_trials)
                clicks.append(n_trials)
                purchases.append(ts_successes)
                returns.append(ts_class_returns)

            return_ts_dequy[index].append(ts_dicty)
            access_count[index] += ts_daily['clicks']
            # ts_rewards.append(ts_daily['reward'])
            update_splitting_table(pricing_learner=pricing_learner, returns=returns, clicks=clicks, purchases=purchases,
                                   pricing_pulled_arm=pricing_pulled_arms[index])
            pricing_learner.update_poisson_context(return_ts_dequy[index].popleft())

            if check_convergence(return_arms[index])[0]:
                Didi = {'arm': check_convergence(return_arms[index])[1], 'day': d, 'exp': e}
                # conv_arms.append(Didi)
                if exp_params['first2'][index] or d == (exp_params['T'] - 1):
                    print('\nIndex:', index)
                    print('Convergence price', check_convergence(return_arms[index])[1], ' - ', 'Convergence Day', d)
                exp_params['first2'][index] = False
                if pricing_learner.table.n_classes == 2:
                    split, proposed_prices = pricing_learner.evaluate_means(margins=margins, chosen_arms=arms,
                                                                            list_acc=ts_daily['list_acc'])
                    if exp_params['first2'][index]:
                        print(split, proposed_prices)
                        # exp_params['first2'][index] = True

            if pricing_learner.table.n_classes == 2 and split == 0:
                class0 = return_ts_classes[index][0]
                class1 = return_ts_classes[index][1]
                ts0 = TS_Learner7(n_arms=pr_env.n_arms, names=[class0.name], n_features=0)
                ts0.inherit(pricing_learner, split, 0)
                ts0_dequy = adapt_dequy(return_ts_dequy[index], [0])
                ts1 = TS_Learner7(n_arms=pr_env.n_arms, names=[class1.name], n_features=0)
                ts1.inherit(pricing_learner, split, 1)
                ts1_dequy = adapt_dequy(return_ts_dequy[index], [1])
                ind_Auxiliary = []
                for indy in range(len(LEARNERS_LIST)):
                    if indy != index:
                        ind_Auxiliary.append(indy)

                LEARNERS_LIST = [ts0, ts1, LEARNERS_LIST[ind_Auxiliary[0]], LEARNERS_LIST[ind_Auxiliary[1]]]
                return_ts_classes = [[class0], [class1], return_ts_classes[ind_Auxiliary[0]],
                                     return_ts_classes[ind_Auxiliary[1]]]
                return_ts_dequy = [ts0_dequy, ts1_dequy, return_ts_dequy[ind_Auxiliary[0]],
                                   return_ts_dequy[ind_Auxiliary[1]]]
                return_bid_environment = [BiddingEnvironment6(bids, acc_sigma=bidding_params['acc_sigma'],
                                                              cost_sigma=bidding_params['cost_sigma'],
                                                              user_classes=[class0],
                                                              n_arms=exp_params['n_arms']),
                                          BiddingEnvironment6(bids, acc_sigma=bidding_params['acc_sigma'],
                                                              cost_sigma=bidding_params['cost_sigma'],
                                                              user_classes=[class1],
                                                              n_arms=exp_params['n_arms']),
                                          return_bid_environment[ind_Auxiliary[0]],
                                          return_bid_environment[ind_Auxiliary[1]]
                                          ]
                return_arms = [[], [], return_arms[ind_Auxiliary[0]], return_arms[ind_Auxiliary[1]]]
                break

        rewards_c3.append(compl_reward)
        if check_convergence(bids_p, crit=0.7)[0] and first:
            first = False
            print('\nConv bid', check_convergence(bids_p, crit=0.7)[1], 'Convergence Day', d)

        costpc = np.sum(cpc_array) / np.sum(access_count)
        bidding_learner.update(pulled_arm=bidding_pulled_arm, costs=costpc, clicks=np.sum(access_count))
        if len(LEARNERS_LIST) != 3:
            break

    if starting_point == (exp_params['T'] - 1): starting_point += 1

    # FOURTH CYCLE
    print('FOURTH CYCLE')
    rewards_c4 = []
    for d in range(starting_point, exp_params['T']):
        starting_point = d+1


        pricing_pulled_arms = []
        compl_reward = 0
        for index, pricing_learner in enumerate(LEARNERS_LIST):
            arm = pricing_learner.pull_arm(margins)
            pricing_pulled_arms.append(arm)
            return_arms[index].append(arm)

        bidding_pulled_arm = bidding_learner.pull_arm_context(learners_list=LEARNERS_LIST,
                                                              price_idx_list=pricing_pulled_arms,
                                                              margins=[margin(prices[p]) for p in pricing_pulled_arms],
                                                              classes_list=return_ts_classes)
        bids_p.append(bidding_pulled_arm)
        access_count = np.zeros(shape=len(LEARNERS_LIST))
        cpc_array = np.zeros(shape=len(LEARNERS_LIST))
        for index, pricing_learner in enumerate(LEARNERS_LIST):

            ts_daily = {'reward': 0, 'successes': 0, 'returns': 0, 'clicks': 0, 'list_acc': []}
            costpc = 0
            clicks, purchases, returns = [], [], []
            ts_dicty = {'arm': pricing_pulled_arms[index],
                        'average_returns': np.zeros(shape=len(return_ts_classes[index])),
                        'sample': np.zeros(shape=len(return_ts_classes[index]))}

            for i, c in enumerate(return_ts_classes[index]):

                n_trials, costpc = return_bid_environment[index].round(pulled_arm=bidding_pulled_arm, user_c=i)
                c.accesses = n_trials
                cpc_array[index] += costpc * n_trials
                # quanti acquisti
                ts_successes, ts_class_returns = pr_env.round(pricing_pulled_arms[index], c, n_trials)
                # aggiorno beta
                pricing_learner.update(pricing_pulled_arms[index], ts_successes, n_trials)

                # aggiorno reward_giornaliera
                try:
                    avi = ts_class_returns / ts_successes
                except ZeroDivisionError:
                    avi = 0
                ts_dicty['average_returns'][i] = avi
                ts_dicty['sample'][i] = ts_successes
                # ts_daily['reward'] += ts_reward
                compl_reward += margin(prices[pricing_pulled_arms[index]]) * (
                        ts_successes + ts_class_returns) - n_trials * costpc
                ts_daily['successes'] += ts_successes
                ts_daily['returns'] += ts_class_returns
                ts_daily['clicks'] += n_trials
                ts_daily['list_acc'].append(n_trials)
                clicks.append(n_trials)
                purchases.append(ts_successes)
                returns.append(ts_class_returns)

            return_ts_dequy[index].append(ts_dicty)
            access_count[index] += ts_daily['clicks']
            # ts_rewards.append(ts_daily['reward'])
            update_splitting_table(pricing_learner=pricing_learner, returns=returns, clicks=clicks, purchases=purchases,
                                   pricing_pulled_arm=pricing_pulled_arms[index])
            pricing_learner.update_poisson_context(return_ts_dequy[index].popleft())

            if check_convergence(return_arms[index])[0]:
                Didi = {'arm': check_convergence(return_arms[index])[1], 'day': d, 'exp': e}
                # conv_arms.append(Didi)
                if exp_params['first2'][index] or d == (exp_params['T'] - 1):
                    print('\nIndex:', index)
                    print('Convergence price', check_convergence(return_arms[index])[1], ' - ', 'Convergence Day', d)
                exp_params['first2'][index] = False
                if pricing_learner.table.n_classes == 2:
                    split, proposed_prices = pricing_learner.evaluate_means(margins=margins, chosen_arms=arms,
                                                                            list_acc=ts_daily['list_acc'])
                    if exp_params['first2'][index]:
                        print(split, proposed_prices)
                        # exp_params['first2'][index] = True

        rewards_c4.append(compl_reward)

        if check_convergence(bids_p, crit=0.7)[0] and first:
            first = False
            print('\nConv bid', check_convergence(bids_p, crit=0.7)[1], 'Convergence Day', d)

        costpc = np.sum(cpc_array) / np.sum(access_count)
        bidding_learner.update(pulled_arm=bidding_pulled_arm, costs=costpc, clicks=np.sum(access_count))

    final_bids.append(bids_p)
    print(len(bids_p))
    lengths.append(len(bids_p))
    print('Number of classes proposed: ', len(LEARNERS_LIST))
    noc.append(len(LEARNERS_LIST))
    # print(arms)
    # ts_rewards.insert(0, 0)
    # ts_final_rewards.append(ts_rewards)
    full_cycle_reward = rewards_c1 + rewards_c2 + rewards_c3 + rewards_c4
    if True or len(full_cycle_reward) == exp_params['T']:
        ts_final_rewards.append(np.array(full_cycle_reward))

ts_final_rewards = np.array(ts_final_rewards)
final_bids = np.array(final_bids)
l_av = np.mean(np.array(lengths))
print(l_av)

plt.figure(0)
plot_regret(opt, [ts_final_rewards], names=['TS_Learner_All_Included'], color_list=['r'])
plt.figure(1)
plot_reward(opt, [ts_final_rewards], names=['TS_Learner_All_Included'], color_list=['r'])

vettore = np.zeros(shape=exp_params['n_arms'])
for i, b in enumerate(bids):
    for c in classes:
        vettore[i] += c.clicks(b)
