from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from ope import SLOPE
from ope import InversePropensityScoreForRanking as IPS
from ope.importance_weight import adaptive_weight
from ope.importance_weight import marginalized_weight
from ope.importance_weight import marginalized_weight_hat
from ope.importance_weight import vanilla_weight
from utils import MARGINALIZED_ESTIMATORS_HAT_TO_BEHAVIOR
from utils import MARGINALIZED_ESTIMATORS_TO_BEHAVIOR
from utils import TRUE_ADAPTIVE_ESTIMATORS_TO_BEHAVIOR
from utils import VANILLA_ESTIMATORS_TO_BEHAVIOR


@dataclass
class RankingOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: list[IPS]
    ope_estimators_tune: Optional[list[SLOPE]] = None
    alpha: Optional[np.ndarray] = None
    pi_a_x_e_estimator: ClassifierMixin = RandomForestClassifier(n_estimators=20, max_depth=10)

    def __post_init__(self) -> None:
        self.estimator_names = {estimator.estimator_name for estimator in self.ope_estimators}

        if self.alpha is None:
            self.alpha = np.ones(self.bandit_feedback["len_list"])

    def _create_estimator_inputs(self, action_dist: np.ndarray) -> dict:
        if len(self.estimator_names & {"MSIPS", "MIIPS", "MRIPS"}) >= 1:
            w_x_e_k = marginalized_weight(
                data=self.bandit_feedback, action_dist=action_dist, behavior_assumption="independent"
            )

        if len(self.estimator_names & {r"MSIPS-$\hat{w}$", r"MIIPS-$\hat{w}$", r"MRIPS-$\hat{w}$"}) >= 1:
            w_hat_x_e_k = marginalized_weight_hat(
                data=self.bandit_feedback,
                action_dist=action_dist,
                pi_a_x_e_estimator=self.pi_a_x_e_estimator,
                behavior_assumption="independent",
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

            elif estimator.estimator_name in MARGINALIZED_ESTIMATORS_TO_BEHAVIOR:
                weight_ = marginalized_weight(
                    data=self.bandit_feedback,
                    action_dist=action_dist,
                    w_x_e_k=w_x_e_k,
                    behavior_assumption=estimator.behavior_assumption,
                )

            elif estimator.estimator_name in MARGINALIZED_ESTIMATORS_HAT_TO_BEHAVIOR:
                weight_ = marginalized_weight_hat(
                    data=self.bandit_feedback,
                    action_dist=action_dist,
                    w_hat_x_e_k=w_hat_x_e_k,
                    behavior_assumption=estimator.behavior_assumption,
                )
            else:
                raise NotImplementedError

            input_data[estimator.estimator_name] = {
                "weight": weight_,
                "alpha": self.alpha,
                "reward": self.bandit_feedback["reward"],
            }

        return input_data

    def estimate_policy_values(self, action_dist: np.ndarray) -> dict:
        input_data = self._create_estimator_inputs(action_dist=action_dist)

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(**input_data[estimator.estimator_name])
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        if self.ope_estimators_tune:
            for estimator_tune in self.ope_estimators_tune:
                estimated_policy_value = estimator_tune.estimate_policy_value_with_tune(
                    bandit_feedback=self.bandit_feedback, action_dist=action_dist
                )
                estimated_policy_values[estimator_tune.estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values
