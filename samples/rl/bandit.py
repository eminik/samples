import logging
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from typing import List
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt

from samples.rl.errors import NoBanditsError


logger = logging.getLogger(__name__)


class Bandit:
    def __init__(self, m: float, lower_bound: float = None, upper_bound:float = None):
        """
        Simulates bandit.

        Args:
            m (float): True mean.
            lower_bound (float): Lower bound for rewards.
            upper_bound (float): Upper bound for rewards.
        """

        self.m = m
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.id = uuid4()

    def pull(self):
        """
        Simulate pulling the arm of the bandit.
        Normal distribution with mu = self.m and sigma = 1. If lower_bound or upper_bound are defined then the
        distribution will be truncated (which will affect the mean too).
        """
        n = 10
        possible_rewards = np.random.randn(n) + self.m

        allowed = np.array([True] * n)
        if self.lower_bound is not None:
            allowed = possible_rewards >= self.lower_bound
        if self.upper_bound is not None:
            allowed *= possible_rewards <= self.upper_bound

        return possible_rewards[allowed][0]


class BanditRewardsLog:
    def __init__(self):
        self.total_actions = 0
        self.total_rewards = 0
        self.all_rewards = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0))

    def record_action(self, bandit, reward):
        self.total_actions += 1
        self.total_rewards += reward
        self.all_rewards.append(reward)
        self.record[bandit.id]['actions'] += 1
        self.record[bandit.id]['reward'] += reward

    def __getitem__(self, bandit):
        return self.record[bandit.id]


class Agent(ABC):

    def __init__(self):
        self.rewards_log = BanditRewardsLog()
        self._bandits = None

    @property
    def bandits(self) -> List[Bandit]:
        if not self._bandits:
            raise NoBanditsError()
        return self._bandits

    @bandits.setter
    def bandits(self, val: List[Bandit]):
        self._bandits = val

    @abstractmethod
    def take_action(self):
        ...

    def take_actions(self, n: int):
        for _ in range(n):
            self.take_action()


def compare_agents(agents: List[Agent], bandits: List[Bandit], iterations: int, show_plot=True):
    for agent in agents:
        logger.info("Running for agent = %s", agent)
        agent.bandits = bandits
        agent.take_actions(iterations)
        if show_plot:
            plt.plot(np.cumsum(agent.rewards_log.all_rewards), label=str(agent))

    if show_plot:
        plt.xlabel("iteration")
        plt.ylabel("total rewards")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
