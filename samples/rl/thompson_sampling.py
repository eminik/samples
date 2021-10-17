import logging

import numpy as np
from scipy import stats

from samples.rl.bandit import Agent


logger = logging.getLogger(__name__)


EXPECTED_REWARD_DISTRIBUTIONS = ('bernoulli', 'normal')


class BayesianAgent(Agent):
    def __init__(self, reward_distr='bernoulli', known_reward_precision=None, mu_prior_mean=None, mu_prior_precision=None):
        if reward_distr not in EXPECTED_REWARD_DISTRIBUTIONS:
            raise ValueError('reward_distr must be in {}.'.format(EXPECTED_REWARD_DISTRIBUTIONS))

        self.known_reward_precision = (known_reward_precision or 1) if reward_distr == 'normal' else None
        self.mu_prior_mean = mu_prior_mean or 0  if reward_distr == 'normal' else None
        self.mu_prior_precision = mu_prior_precision or 1  if reward_distr == 'normal' else None # 1 / variance

        self.reward_distr = reward_distr
        super().__init__()

    @staticmethod
    def _sample_beta(bandit_record):
        # + 1 for a Beta(1, 1) prior
        successes = bandit_record['reward'] + 1
        failures = bandit_record['actions'] - bandit_record['reward'] + 1
        return np.random.beta(a=successes, b=failures, size=1)[0]

    def _sample_normal(self, bandit_record):
        reward_precision = self.known_reward_precision # 1 / variance

        mu_posterior_precision = self.mu_prior_precision + reward_precision * bandit_record['actions']
        mu_posterior_mean = (
                self.mu_prior_mean * self.mu_prior_precision + reward_precision * bandit_record['reward']
        ) / mu_posterior_precision
        mu_posterior_std = np.sqrt(1 / mu_posterior_precision)

        return np.random.normal(loc=mu_posterior_mean, scale=mu_posterior_std, size=1)[0]

    def _sample_bandit_mean(self, bandit):
        bandit_record = self.rewards_log[bandit]

        if self.reward_distr == 'bernoulli':
            return self._sample_beta(bandit_record)
        elif self.reward_distr == 'normal':
            return self._sample_normal(bandit_record)
        else:
            raise NotImplementedError()

    def get_bandit_distr(self, bandit):
        bandit_record = self.rewards_log[bandit]
        if self.reward_distr == 'bernoulli':
            alpha = bandit_record['reward'] + 1
            beta = bandit_record['actions'] - bandit_record['reward'] + 1

            x = np.linspace(0, 1.0, 100)
            y = stats.beta.pdf(x, alpha, beta)
        else:
            reward_precision = self.known_reward_precision

            mu_posterior_precision = self.mu_prior_precision + reward_precision * bandit_record['actions']
            mu_posterior_mean = (self.mu_prior_mean * self.mu_prior_precision + reward_precision * bandit_record[
                'reward']) / mu_posterior_precision
            mu_posterior_std = np.sqrt(1 / mu_posterior_precision)

            x = np.linspace(0, 20.0, 100)  # this should ideally be a variable
            y = stats.norm.pdf(x, loc=mu_posterior_mean, scale=mu_posterior_std)

        return x, y

    def take_action(self):
        samples = [self._sample_bandit_mean(bandit) for bandit in self.bandits]
        current_bandit = self.bandits[np.argmax(samples)]
        reward = current_bandit.pull()
        self.rewards_log.record_action(current_bandit, reward)
        return reward

    def __repr__(self):
        return 'BayesianAgent(reward_distr="{}")'.format(self.reward_distr)
