from dataclasses import dataclass
from typing import Optional

import numpy as np

from ope import EmbeddingSelectionWithSLOPE
from ope import InversePropensityScoreForRanking as IPS
from ope.importance_weight import adaptive_weight
from ope.importance_weight import marginalized_weight
from ope.importance_weight import vanilla_weight
from utils import TRUE_ADAPTIVE_ESTIMATORS_TO_BEHAVIOR
from utils import TRUE_MARGINALIZED_ESTIMATORS_TO_BEHAVIOR
from utils import VANILLA_ESTIMATORS_TO_BEHAVIOR


@dataclass
class RankingOffPolicyEvaluation:
    """Base class for off-policy evaluation for ranking.

    Args:
        bandit_feedback: dict
            bandit feedback data collected by a behavior policy.

        ope_estimators: list[IPS]
            list of OPE estimators.

        ope_estimators_tune: Optional[list[SLOPE]] = None
            list of OPE estimators with hyperparameter tuning.

        alpha: Optional[np.ndarray] = None
            alpha parameter for rankings.

        pi_a_x_e_estimator: ClassifierMixin = CategoricalNB()
            a classifier to estimate the conditional distribution of actions given context.
    """

    bandit_feedback: dict
    ope_estimators: list[IPS]
    alpha: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.estimator_names = {estimator.estimator_name for estimator in self.ope_estimators}

        if self.alpha is None:
            self.alpha = np.ones(self.bandit_feedback["len_list"])

    def _create_estimator_inputs(self, action_dist: np.ndarray) -> dict:
        """Create input data for OPE estimators.

        Args:
            action_dist: np.ndarray
                action distribution of evaluation policy

        Returns:
            dict: input data for OPE estimators.
        """

        if len(self.estimator_names & {"MSIPS", "MIIPS", "MRIPS"}) >= 1:
            w_x_e_k = marginalized_weight(
                data=self.bandit_feedback, action_dist=action_dist, behavior_assumption="independent"
            )

        input_data = dict()
        for estimator in self.ope_estimators:
            if estimator.estimator_name in TRUE_ADAPTIVE_ESTIMATORS_TO_BEHAVIOR:
                weight_ = adaptive_weight(data=self.bandit_feedback, action_dist=action_dist)

            elif estimator.estimator_name in VANILLA_ESTIMATORS_TO_BEHAVIOR:
                weight_ = vanilla_weight(
                    data=self.bandit_feedback,
                    action_dist=action_dist,
                    behavior_assumption=estimator.behavior_assumption,
                )

            elif estimator.estimator_name in TRUE_MARGINALIZED_ESTIMATORS_TO_BEHAVIOR:
                weight_ = marginalized_weight(
                    data=self.bandit_feedback,
                    action_dist=action_dist,
                    w_x_e_k=w_x_e_k,
                    behavior_assumption=estimator.behavior_assumption,
                )

            else:
                raise NotImplementedError(f"{estimator.estimator_name} is not implemented.")

            input_data[estimator.estimator_name] = {
                "weight": weight_,
                "alpha": self.alpha,
                "reward": self.bandit_feedback["reward"],
            }

        return input_data

    def estimate_policy_values(self, action_dist: np.ndarray) -> dict:
        """Estimate the policy values of evaluation policy.

        Args:
            action_dist: np.ndarray
                action distribution of evaluation policy

        Returns:
            dict: estimated policy values of evaluation policy for each estimators.
        """

        input_data = self._create_estimator_inputs(action_dist=action_dist)

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(**input_data[estimator.estimator_name])
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values


@dataclass
class RankingOffPolicyEvaluationWithTune:
    """Base class for off-policy evaluation for ranking with hyperparameter tuning.

    Args:
        ope_estimators_tune: list[NNAbstractionLearnerWithSLOPE]
            list of OPE estimators with hyperparameter tuning.
    """

    ope_estimators_tune: list[EmbeddingSelectionWithSLOPE]

    def __post_init__(self) -> None:
        self.is_exist_best_param = False

    def estimate_policy_values_with_tune(self, bandit_feedback: dict, action_dist: np.ndarray) -> dict:
        """Estimate the policy values of evaluation policy with hyperparameter tuning.

        Args:
            bandit_feedback: dict
                bandit feedback data collected by a behavior policy.

            action_dist: np.ndarray
                action distribution of evaluation policy

        Returns:
            dict: estimated policy values of evaluation policy for each estimators.
        """

        estimated_policy_values = dict()
        for estimator_tune in self.ope_estimators_tune:
            estimated_policy_value = estimator_tune.estimate_policy_value_with_tune(
                bandit_feedback=bandit_feedback, action_dist=action_dist
            )
            estimated_policy_values[estimator_tune.estimator.estimator_name] = estimated_policy_value

        self.is_exist_best_param = True

        return estimated_policy_values

    def estimate_policy_values_with_best_param(self, bandit_feedback: dict, action_dist: np.ndarray) -> dict:
        """Estimate the policy values of evaluation policy with the best hyperparameters.

        Args:
            bandit_feedback: dict
                bandit feedback data collected by a behavior policy.

            action_dist: np.ndarray
                action distribution of evaluation policy

        Returns:
            dict: estimated policy values of evaluation policy for each estimators.
        """

        if not self.is_exist_best_param:
            raise ValueError("The best hyperparameters have not been found yet.")

        estimated_policy_values = dict()
        for estimator_tune in self.ope_estimators_tune:
            estimated_policy_value = estimator_tune.estimate_policy_value_with_best_param(
                bandit_feedback=bandit_feedback, action_dist=action_dist
            )
            estimated_policy_values[estimator_tune.estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values
