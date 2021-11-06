from finetuning.finetuning_alltogether import *

opt = 19650.95
acc_list = [10, 50]
cost_list = [0.05, 0.1]
conf_list = [20, 50]
rw_list = [1]

loe = []
total_i = 1
total_len = len(acc_list) * len(cost_list) * len(conf_list) * len(rw_list)
for acc_sigma in acc_list:
    for cost_sigma in cost_list:
        for conf_coef in conf_list:
            for rw_coef in rw_list:
                print('exp: ', total_i, 'out of: ', total_len)
                cl, ncl, cumulative = allTogether(acc_sigma=acc_sigma, cost_sigma=cost_sigma, T=365, noe=25,
                                                  conf_coef=conf_coef, rw_coef=rw_coef, opt=opt)
                loe.append({'acc_sigma': acc_sigma, 'cost_sigma': cost_sigma, 'conf_coef': conf_coef, 'cl': cl,
                            'ncl': ncl, 'cumulative regret': cumulative})
                total_i += 1

for el in loe:
    print(el)

