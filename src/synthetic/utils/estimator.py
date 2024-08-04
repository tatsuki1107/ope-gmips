from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseSlateInversePropensityScore(metaclass=ABCMeta):
    estimator_name: str
    pscore_type: str

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self.estimate_round_rewards(
            reward=reward,
            behavior_policy_pscore=behavior_policy_pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        ).mean()

    @abstractmethod
    def estimate_round_rewards(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SIPS(BaseSlateInversePropensityScore):
    """Standard Inverse Propensity Score Estimator Class."""

    def estimate_round_rewards(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> np.ndarray:
        weight = evaluation_policy_pscore / behavior_policy_pscore
        rank_weight = weight.prod(1)

        return rank_weight * reward.sum(1)


@dataclass
class SNSIPS(SIPS):
    """Self Normalized Standard Inverse Propensity Score Estimator Class."""

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> float:
        return (
            self.estimate_round_rewards(
                reward=reward,
                behavior_policy_pscore=behavior_policy_pscore,
                evaluation_policy_pscore=evaluation_policy_pscore,
            ).sum()
            / (evaluation_policy_pscore / behavior_policy_pscore).sum()
        )


@dataclass
class IIPS(BaseSlateInversePropensityScore):
    """Independent Inverse Propensity Score Estimator Class."""

    def estimate_round_rewards(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> np.ndarray:
        position_weight = evaluation_policy_pscore / behavior_policy_pscore

        return (position_weight * reward).sum(1)


@dataclass
class RIPS(BaseSlateInversePropensityScore):
    """Reward-Interaction Inverse Propensity Score Estimator Class."""

    def estimate_round_rewards(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> np.ndarray:
        weight = evaluation_policy_pscore / behavior_policy_pscore
        K = reward.shape[1]

        top_k_weight = []
        for k in range(K):
            top_k_weight.append(weight[:, : k + 1].prod(axis=1, keepdims=True))

        top_k_weight = np.concatenate(top_k_weight, axis=1)

        return (top_k_weight * reward).sum(1)


@dataclass
class AIPS(BaseSlateInversePropensityScore):
    def __post_init__(self) -> None:
        self.behavior_to_estimator_dict: dict[str, BaseSlateInversePropensityScore] = {
            "all": SIPS(estimator_name="SIPS", pscore_type=self.pscore_type),
            "independent": IIPS(estimator_name="IIPS", pscore_type=self.pscore_type),
            "cascade": RIPS(estimator_name="RIPS", pscore_type=self.pscore_type),
        }

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        user_behavior: np.ndarray,
    ) -> float:
        return self.estimate_round_rewards(
            reward=reward,
            behavior_policy_pscore=behavior_policy_pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
            user_behavior=user_behavior,
        ).mean()

    def estimate_round_rewards(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        user_behavior: np.ndarray,
    ) -> np.ndarray:
        estimated_round_rewards = []
        for behavior, estimator in self.behavior_to_estimator_dict.items():
            c = user_behavior == behavior
            estimated_round_rewards.extend(
                estimator.estimate_round_rewards(
                    reward=reward[c],
                    behavior_policy_pscore=behavior_policy_pscore[c],
                    evaluation_policy_pscore=evaluation_policy_pscore[c],
                ).tolist()
            )

        return np.array(estimated_round_rewards)


@dataclass
class SlateOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: list[BaseSlateInversePropensityScore]

    def _create_estimator_inputs(
        self,
        evaluation_policy_pscore_dict: dict,
    ) -> dict:
        input_data = {}
        for estimator in self.ope_estimators:
            behavior_policy_pscore = self.bandit_feedback["pscore"][estimator.pscore_type]
            evaluation_policy_pscore = evaluation_policy_pscore_dict[estimator.pscore_type]

            input_data[estimator.estimator_name] = {
                "reward": self.bandit_feedback["reward"],
                "behavior_policy_pscore": behavior_policy_pscore,
                "evaluation_policy_pscore": evaluation_policy_pscore,
            }

            if "AIPS" in estimator.estimator_name:
                input_data[estimator.estimator_name]["user_behavior"] = self.bandit_feedback["user_behavior"]

        return input_data

    def estimate_policy_values(
        self,
        evaluation_policy_pscore_dict: dict,
    ) -> dict:
        input_data = self._create_estimator_inputs(evaluation_policy_pscore_dict=evaluation_policy_pscore_dict)

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(**input_data[estimator.estimator_name])
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values
