import logging

import numpy as np
from scipy import stats

from samples.rl.bandit import (
    Agent,
    Bandit,
)


logger = logging.getLogger(__name__)


class BayesianAgent(Agent):
    def __init__(self, reward_distr='bernoulli'):
        if reward_distr not in ('bernoulli'):
            raise ValueError('reward_distr must be "bernoulli".')

        self.reward_distr = reward_distr
        super().__init__()

    def _sample_bandit_mean(self, bandit):
        bandit_record = self.rewards_log[bandit]

        if self.reward_distr == 'bernoulli':
            # + 1 for a Beta(1, 1) prior
            successes = bandit_record['reward'] + 1
            failures = bandit_record['actions'] - bandit_record['reward'] + 1
            return np.random.beta(a=successes, b=failures, size=1)
        else:
            raise NotImplementedError()

    def get_bandit_distr(self, bandit):
        bandit_record = self.rewards_log[bandit]
        if self.reward_distr == 'bernoulli':
            alpha = bandit_record['reward'] + 1
            beta = bandit_record['actions'] - bandit_record['reward'] + 1

            x = np.linspace(0, 1.0, 100)
            y = stats.beta.pdf(x, alpha, beta)
            return x, y
        else:
            raise NotImplementedError()

    def take_action(self):
        samples = [self._sample_bandit_mean(bandit) for bandit in self.bandits]
        current_bandit = self.bandits[np.argmax(samples)]
        reward = current_bandit.pull()
        self.rewards_log.record_action(current_bandit, reward)
        return reward

    def __repr__(self):
        return 'BayesianAgent(reward_distr="{}")'.format(self.reward_distr)
