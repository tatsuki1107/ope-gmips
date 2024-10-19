from dataclasses import dataclass
from typing import Callable

import numpy as np
from obp.ope import BaseOffPolicyEstimator

from utils import MARGINALIZED_ESTIMATORS_TO_BEHAVIOR
from utils import VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR
from utils import VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR


@dataclass
class InversePropensityScoreForRanking(BaseOffPolicyEstimator):
    """Base class for off-policy estimators based on the inverse propensity score for ranking.

    Args:
        estimator_name: str
            name of the estimator.
    """

    estimator_name: str

    def __post_init__(self) -> None:
        if self.estimator_name not in VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR:
            raise ValueError(f"estimator_name must be one of {VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR.keys()}")

        self.behavior_assumption = VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR[self.estimator_name]

    def _estimate_round_rewards(self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Args:
            reward: np.ndarray
                reward matrix of shape (n_rounds, len_list)
            alpha: np.ndarray
                action choice matrix of shape (n_rounds, len_list)
            weight: np.ndarray
                importance weight matrix of shape (n_rounds, len_list)

        Returns:
            np.ndarray: estimated round-wise rewards.
        """
        return (weight * alpha * reward).sum(1)

    def estimate_policy_value(self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.float64:
        """Estimate the policy value of evaluation policy.

        Args:
            reward: np.ndarray
                reward matrix of shape (n_rounds, len_list)
            alpha: np.ndarray
                action choice matrix of shape (n_rounds, len_list)
            weight: np.ndarray
                importance weight matrix of shape (n_rounds, len_list)

        Returns:
            np.float64: estimated policy value.
        """

        return self._estimate_round_rewards(reward=reward, alpha=alpha, weight=weight).mean()

    def estimate_interval(self):
        pass


@dataclass
class SelfNormalizedIPSForRanking(InversePropensityScoreForRanking):
    """Self-Normalized Inverse Propensity Score for Ranking.

    References:
        Swaminathan, Adith, and Thorsten Joachims.
        "The self-normalized estimator for counterfactual learning.", 2015.
    """

    def __post_init__(self) -> None:
        if self.estimator_name not in VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR:
            raise ValueError(f"estimator_name must be one of {VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR.keys()}")

        self.behavior_assumption = VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR[self.estimator_name]

    def _estimate_round_rewards(self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Args:
            reward: np.ndarray
                reward matrix of shape (n_rounds, len_list)

            alpha: np.ndarray
                action choice matrix of shape (n_rounds, len_list)

            weight: np.ndarray
                importance weight matrix of shape (n_rounds, len_list)

        Returns:
            np.ndarray: estimated round-wise rewards.
        """

        return (weight * alpha * reward / weight.mean(0)).sum(1)


@dataclass
class MarginalizedIPSForRanking(InversePropensityScoreForRanking):
    """Marginalized Inverse Propensity Score for Ranking.

    References:
        Saito, Yuta, and Thorsten Joachims.
        "Off-policy evaluation for large action spaces via embeddings.", 2022.
    """

    def __post_init__(self) -> None:
        if self.estimator_name not in MARGINALIZED_ESTIMATORS_TO_BEHAVIOR:
            raise ValueError(f"estimator_name must be one of {MARGINALIZED_ESTIMATORS_TO_BEHAVIOR.keys()}")

        self.behavior_assumption = MARGINALIZED_ESTIMATORS_TO_BEHAVIOR[self.estimator_name]

    def estimate_policy_value_with_dev(
        self, reward: np.ndarray, alpha: np.ndarray, weight: np.ndarray, lower_bound_func: Callable, delta: float = 0.05
    ) -> tuple[np.float64]:
        """Estimate the policy value of evaluation policy with confidence interval.

        Args:
            reward: np.ndarray
                reward matrix of shape (n_rounds, len_list)

            alpha: np.ndarray
                action choice matrix of shape (n_rounds, len_list)

            weight: np.ndarray
                marginalized importance weight matrix by action embeddings of shape (n_rounds, len_list)

            lower_bound_func: Callable
                function to compute the lower bound of the confidence interval.

            delta: float = 0.05
                confidence level.

        Returns:
            tuple[np.float64]: estimated policy value and confidence interval.
        """

        r_hat = self._estimate_round_rewards(reward=reward, alpha=alpha, weight=weight)
        cnf = lower_bound_func(r_hat, delta=delta, with_dev=True)
        return np.mean(r_hat), cnf
