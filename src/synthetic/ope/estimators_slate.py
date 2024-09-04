from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from obp.ope import BaseOffPolicyEstimator


@dataclass
class BaseSlateInversePropensityScore(BaseOffPolicyEstimator):
    estimator_name: str
    pscore_type: str

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(reward=reward, position_wise_weight=position_wise_weight).mean()

    @abstractmethod
    def _estimate_round_rewards(self, reward: np.ndarray, position_wise_weight: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def estimate_interval(self):
        pass


@dataclass
class StandardIPS(BaseSlateInversePropensityScore):
    """Standard Inverse Propensity Score Estimator Class."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> np.ndarray:
        ranking_wise_weight = position_wise_weight.prod(1)

        return ranking_wise_weight * reward.sum(1)


@dataclass
class SelfNormalizedStandardIPS(BaseSlateInversePropensityScore):
    """Self Normalized Standard Inverse Propensity Score Estimator Class."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> np.ndarray:
        ranking_wise_weight = position_wise_weight.prod(1)

        return (ranking_wise_weight / ranking_wise_weight.mean()) * reward.sum(1)


@dataclass
class IndependentIPS(BaseSlateInversePropensityScore):
    """Independent Inverse Propensity Score Estimator Class."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> np.ndarray:
        return (position_wise_weight * reward).sum(1)


@dataclass
class SelfNormalizedIndependentIPS(BaseSlateInversePropensityScore):
    """Self Normalized Independent Inverse Propensity Score Estimator Class."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> np.ndarray:
        return (position_wise_weight * reward / position_wise_weight.mean(0)).sum(1)


@dataclass
class RewardInteractionIPS(BaseSlateInversePropensityScore):
    """Reward-Interaction Inverse Propensity Score Estimator Class."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> np.ndarray:
        K = reward.shape[1]

        top_k_weight = []
        for k in range(K):
            top_k_weight.append(position_wise_weight[:, : k + 1].prod(axis=1, keepdims=True))

        top_k_weight = np.concatenate(top_k_weight, axis=1)

        return (top_k_weight * reward).sum(1)


@dataclass
class SelfNormalizedRewardInteractionIPS(BaseSlateInversePropensityScore):
    """Self Normalized Reward-Interaction Inverse Propensity Score Estimator Class."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
    ) -> np.ndarray:
        K = reward.shape[1]

        top_k_weight = []
        for k in range(K):
            top_k_weight.append(position_wise_weight[:, : k + 1].prod(axis=1, keepdims=True))

        top_k_weight = np.concatenate(top_k_weight, axis=1)

        return (top_k_weight * reward / top_k_weight.mean(0)).sum(1)


@dataclass
class AdaptiveIPS(BaseSlateInversePropensityScore):
    def __post_init__(self) -> None:
        self.behavior_index_to_estimator_dict: dict[str, BaseSlateInversePropensityScore] = {
            "standard": StandardIPS(estimator_name="SIPS", pscore_type=self.pscore_type),
            "independent": IndependentIPS(estimator_name="IIPS", pscore_type=self.pscore_type),
            "cascade": RewardInteractionIPS(estimator_name="RIPS", pscore_type=self.pscore_type),
        }

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        position_wise_weight,
        user_behavior: np.ndarray,
    ) -> float:
        return self._estimate_round_rewards(
            reward=reward,
            position_wise_weight=position_wise_weight,
            user_behavior=user_behavior,
        ).mean()

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position_wise_weight: np.ndarray,
        user_behavior: np.ndarray,
    ) -> np.ndarray:
        estimated_round_rewards = []
        for behavior_, estimator in self.behavior_index_to_estimator_dict.items():
            c = user_behavior == behavior_
            if np.any(c):
                estimated_round_rewards.extend(
                    estimator._estimate_round_rewards(
                        reward=reward[c],
                        position_wise_weight=position_wise_weight[c],
                    ).tolist()
                )

        return np.array(estimated_round_rewards)


@dataclass
class SelfNormalizedAdaptiveIPS(AdaptiveIPS):
    def __post_init__(self) -> None:
        self.behavior_index_to_estimator_dict: dict[str, BaseSlateInversePropensityScore] = {
            "standard": SelfNormalizedStandardIPS(estimator_name="snSIPS", pscore_type=self.pscore_type),
            "independent": SelfNormalizedIndependentIPS(estimator_name="snIIPS", pscore_type=self.pscore_type),
            "cascade": SelfNormalizedRewardInteractionIPS(estimator_name="snRIPS", pscore_type=self.pscore_type),
        }
