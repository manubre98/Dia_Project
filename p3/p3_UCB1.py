from .p3_Learner import *


class UCB1(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.number_pulled = np.zeros(n_arms)
        self.first_round = 0

    def pull_arm(self, margins):
        if self.first_round < self.n_arms:
            arm = self.first_round
        else:
            means = self.poisson_vector[:, 0] + np.ones(shape=self.poisson_vector.shape[0])
            # upper_bound = (self.empirical_means + self.confidence) * means * margins
            upper_bound = (self.empirical_means + self.confidence) * means * margins
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update(self, pulled_arm, successes, n_trials):
        self.t += n_trials
        if self.first_round <= self.n_arms:
            self.first_round += 1

        self.number_pulled[pulled_arm] += n_trials
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] *
                                            (self.number_pulled[pulled_arm] - n_trials) + successes) \
                                           / self.number_pulled[pulled_arm]
        for a in range(self.n_arms):
            number_pulled = max(1, self.number_pulled[a])
            self.confidence[a] = (2 * np.log(self.t) / number_pulled) ** 0.5
