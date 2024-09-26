from dataclasses import dataclass
from typing import Callable

import numpy as np
from obp.ope import BaseOffPolicyEstimator

from utils import MARGINALIZED_ESTIMATORS_TO_BEHAVIOR
from utils import VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR
from utils import VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR


@dataclass
class InversePropensityScoreForRanking(BaseOffPolicyEstimator):
    estimator_name: str

    def __post_init__(self) -> None:
        if self.estimator_name not in VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR:
            raise ValueError

        self.behavior_assumption = VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR[self.estimator_name]

    def _estimate_round_rewards(self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        return (weight * alpha * reward).sum(1)

    def estimate_policy_value(self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.float64:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(reward=reward, alpha=alpha, weight=weight).mean()

    def estimate_interval(self):
        pass


@dataclass
class SelfNormalizedIPSForRanking(InversePropensityScoreForRanking):
    def __post_init__(self) -> None:
        if self.estimator_name not in VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR:
            raise ValueError
        self.behavior_assumption = VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR[self.estimator_name]

    def _estimate_round_rewards(self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return (weight * alpha * reward / weight.mean(0)).sum(1)


@dataclass
class MarginalizedIPSForRanking(InversePropensityScoreForRanking):
    def __post_init__(self) -> None:
        if self.estimator_name not in MARGINALIZED_ESTIMATORS_TO_BEHAVIOR:
            raise ValueError
        self.behavior_assumption = MARGINALIZED_ESTIMATORS_TO_BEHAVIOR[self.estimator_name]

    def estimate_policy_value_with_dev(
        self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray, lower_bound_func: Callable, delta: float = 0.05
    ) -> tuple[np.float64]:
        r_hat = self._estimate_round_rewards(reward=reward, alpha=alpha, weight=weight)
        cnf = lower_bound_func(r_hat, delta=delta, with_dev=True)
        return np.mean(r_hat), cnf
