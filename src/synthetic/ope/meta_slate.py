from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from ope import BaseSlateInversePropensityScore


@dataclass
class SlateOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: list[BaseSlateInversePropensityScore]
    pi_a_x_e_estimator: ClassifierMixin = RandomForestClassifier(n_estimators=20, max_depth=10)
    position_wise_weight_hat: Optional[np.ndarray] = None

    def _create_estimator_inputs(
        self,
        action_dist: np.ndarray,
        evaluation_policy_pscore_dict: dict,
    ) -> dict:
        input_data = {}
        for estimator in self.ope_estimators:
            if estimator.estimator_name in {"MSIPS", "MIIPS", "MRIPS"}:
                if self.position_wise_weight_hat is None:
                    self.position_wise_weight_hat = estimate_w_x_e(
                        bandit_feedback=self.bandit_feedback,
                        action_dist=action_dist,
                        pi_a_x_e_estimator=self.pi_a_x_e_estimator,
                    )
                position_wise_weight = self.position_wise_weight_hat
            else:
                behavior_policy_pscore = self.bandit_feedback["pscore"][estimator.pscore_type]
                evaluation_policy_pscore = evaluation_policy_pscore_dict[estimator.pscore_type]
                position_wise_weight = evaluation_policy_pscore / behavior_policy_pscore

            input_data[estimator.estimator_name] = {
                "reward": self.bandit_feedback["reward"],
                "position_wise_weight": position_wise_weight,
            }

            if "AIPS" in estimator.estimator_name:
                input_data[estimator.estimator_name]["user_behavior"] = self.bandit_feedback["user_behavior"]

        return input_data

    def estimate_policy_values(
        self,
        action_dist: np.ndarray,
        evaluation_policy_pscore_dict: dict,
    ) -> dict:
        input_data = self._create_estimator_inputs(
            action_dist=action_dist, evaluation_policy_pscore_dict=evaluation_policy_pscore_dict
        )

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(**input_data[estimator.estimator_name])
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values


def estimate_w_x_e(bandit_feedback: dict, action_dist: np.ndarray, pi_a_x_e_estimator: ClassifierMixin) -> np.ndarray:
    w_x_a_k = action_dist / bandit_feedback["pi_b_k"]
    w_x_a_k = np.where(w_x_a_k < np.inf, w_x_a_k, 0.0)

    w_hat_x_e_k = []
    for pos_ in range(action_dist.shape[-1]):
        w_hat_x_e = _estimate_position_wise_w_x_e(
            w_x_a=w_x_a_k[:, :, pos_],
            context=bandit_feedback["context"],
            action=bandit_feedback["slate_id_at_k"][:, pos_],
            action_embeds=bandit_feedback["action_context"][:, pos_, :],
            pi_a_x_e_estimator=pi_a_x_e_estimator,
        )
        w_hat_x_e_k.append(w_hat_x_e)

    return np.array(w_hat_x_e_k).T


def _estimate_position_wise_w_x_e(
    w_x_a: np.ndarray,
    context: np.ndarray,
    action: np.ndarray,
    action_embeds: np.ndarray,
    pi_a_x_e_estimator: ClassifierMixin,
) -> np.ndarray:
    n_rounds, n_actions_at_position = w_x_a.shape
    # c = OneHotEncoder(
    #    sparse=False,
    #    drop="first",
    # ).fit_transform(action_embeds)
    x_e = np.c_[context, action_embeds]
    pi_hat_a_x_e = np.zeros((n_rounds, n_actions_at_position))
    pi_a_x_e_estimator.fit(x_e, action)
    pi_hat_a_x_e[:, np.unique(action)] = pi_a_x_e_estimator.predict_proba(x_e)

    w_hat_x_e = (w_x_a * pi_hat_a_x_e).sum(1)
    return w_hat_x_e
