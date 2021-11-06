from p3.curves import *


def compute_optimum_bidding(bids, price, user_classes):
    opt_vec = []
    for i in range(np.size(bids)):
        val = 0
        for c in user_classes:
            val += c.clicks(bids[i]) * (c.conversion_rate(price) * margin(price) * (c.returns()+1) - cost_per_click(bids[i]))
        opt_vec.append(val)

    opt_ind = np.argmax(np.array(opt_vec))
    opt = np.max(np.array(opt_vec))

    return opt, opt_ind, opt_vec