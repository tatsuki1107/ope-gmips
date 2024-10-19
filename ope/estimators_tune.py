from abc import ABCMeta
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import numpy as np
from obp.dataset import BaseBanditDataset
from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state

from ope import MarginalizedIPSForRanking
from ope.importance_weight import NNAbstractionLearner
from ope.importance_weight import adaptive_weight
from ope.importance_weight import vanilla_weight
from policy import gen_eps_greedy
from utils import ADAPTIVE_ESTIMATORS_WITH_UBT_TO_BEHAVIOR
from utils import MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR


@dataclass
class BaseOffPolicyEstimatorWithTune(metaclass=ABCMeta):
    """Base class for off-policy estimators with hyperparameter tuning.

    Args:
        estimator: MarginalizedIPSForRanking
            an instance of the estimator class.

        param_name: str
            name of the hyperparameter to be tuned.
    """

    estimator: MarginalizedIPSForRanking
    param_name: str

    @abstractmethod
    def estimate_policy_value_with_tune(self):
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value_with_best_param(self):
        raise NotImplementedError


@dataclass
class NNAbstractionLearnerWithSLOPE(BaseOffPolicyEstimatorWithTune, NNAbstractionLearner):
    """Neural network abstraction learner with SLOPE estimator class.

    Args:
        hyper_param: np.ndarray
            hyperparameter (the number of category dimension) candidates.

        lower_bound_func: Callable[[np.ndarray, float, bool], np.float64]
            function to compute the lower bound of the estimated policy value.

        weight_func: Callable[[dict, np.ndarray], np.ndarray]
            function to compute the importance weight.

        alpha: Optional[np.ndarray] = None
            alpha parameter for rankings.

        delta: float = 0.05
            confidence level.
    """

    hyper_param: np.ndarray
    lower_bound_func: Callable[[np.ndarray, float, bool], np.float64]
    weight_func: Callable[[dict, np.ndarray], np.ndarray]
    alpha: Optional[np.ndarray] = None
    delta: float = 0.05

    def __post_init__(self) -> None:
        self.best_unique_action_context = None

    def estimate_policy_value_with_tune(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        """Estimate the policy value of evaluation policy with hyperparameter tuning.

        Args:
            bandit_feedback: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.float64: estimated policy value.
        """

        if self.alpha is None:
            self.alpha = np.ones(bandit_feedback["len_list"])

        C = np.sqrt(6) - 1
        unique_action_context_dict = {}
        theta_dict_for_sort, cnf_dict_for_sort = {}, {}
        for id, n_cat_dim in enumerate(self.hyper_param):
            self._init_model(n_cat_dim=n_cat_dim)
            unique_action_context, action_context = self.fit_predict(
                context=bandit_feedback["context"],
                action=bandit_feedback["action"],
                pscore=bandit_feedback["pscore"],
                action_id_at_k=bandit_feedback["action_id_at_k"],
            )
            bandit_feedback["unique_action_context"] = unique_action_context
            bandit_feedback["action_context"] = action_context
            bandit_feedback["observed_cat_dim"] = np.arange(n_cat_dim)

            importance_weight = self.weight_func(
                data=bandit_feedback,
                action_dist=action_dist,
                behavior_assumption=self.estimator.behavior_assumption,
            )

            theta, cnf = self.estimator.estimate_policy_value_with_dev(
                reward=bandit_feedback["reward"],
                alpha=self.alpha,
                weight=importance_weight,
                lower_bound_func=self.lower_bound_func,
                delta=self.delta,
            )

            cnf_dict_for_sort[id] = cnf
            theta_dict_for_sort[id] = theta
            unique_action_context_dict[id] = unique_action_context

        theta_list, cnf_list = [], []
        sorted_idx_list = [i for i, _ in sorted(cnf_dict_for_sort.items(), key=lambda k: k[1], reverse=True)]
        for i, idx in enumerate(sorted_idx_list):
            cnf_i = cnf_dict_for_sort[idx]
            theta_i = theta_dict_for_sort[idx]
            if len(theta_list) < 1:
                theta_list.append(theta_i), cnf_list.append(cnf_i)
            else:
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    best_idx = sorted_idx_list[i - 1]
                    self.best_unique_action_context = unique_action_context_dict[best_idx]
                    return theta_dict_for_sort[best_idx]

        self.best_unique_action_context = unique_action_context_dict[sorted_idx_list[-1]]
        return theta_dict_for_sort[sorted_idx_list[-1]]

    def estimate_policy_value_with_best_param(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        """Estimate the policy value of evaluation policy with the best hyperparameter.

        Args:
            bandit_feedback: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy

        Returns:
            np.float64: estimated policy value.
        """

        if self.best_unique_action_context is None:
            raise ValueError(
                "best_unique_action_context is not found. Please run estimate_policy_value_with_tune method."
            )

        bandit_feedback["unique_action_context"] = self.best_unique_action_context
        bandit_feedback["action_context"] = self.best_unique_action_context[
            np.arange(bandit_feedback["len_list"])[None, :], bandit_feedback["action_id_at_k"]
        ]

        bandit_feedback["observed_cat_dim"] = np.arange(self.best_unique_action_context.shape[2])
        importance_weight = self.weight_func(
            data=bandit_feedback,
            action_dist=action_dist,
            behavior_assumption=self.estimator.behavior_assumption,
        )

        estimated_value = self.estimator.estimate_policy_value(
            reward=bandit_feedback["reward"],
            alpha=self.alpha,
            weight=importance_weight,
        )

        return estimated_value


@dataclass
class EmbeddingSelectionWithSLOPE(BaseOffPolicyEstimatorWithTune):
    """Selection by Lepski's principle for Off-Policy Evaluation (SLOPE) estimator class.

    Args:
        hyper_param: np.ndarray
            hyperparameter candidates.

        lower_bound_func: Callable[[np.ndarray, float, bool], np.float64]
            function to compute the lower bound of the policy value.

        weight_func: Callable[[dict, np.ndarray], np.ndarray]
            function to compute the importance weight.

        tuning_method: str
            method to tune the hyperparameter.

        weight_estimator: Optional[ClassifierMixin]
            an instance of the classifier to estimate the importance weight.

        alpha: Optional[np.ndarray]
            action choice matrix of shape (n_rounds, len_list).

        min_combination: int
            minimum number of combinations of the hyperparameters.

        delta: float
            confidence level.

    References:
        Su, Yi, Pavithra Srinath, and Akshay Krishnamurthy.
        "Adaptive estimator selection for off-policy evaluation.", 2020.

        Tucker, George, and Jonathan Lee.
        "Improved estimator selection for off-policy evaluation.", 2021.

        Yuta Saito, Thorsten Joachims.
        "Off-policy evaluation for large action spaces via embeddings.", 2022.
    """

    hyper_param: np.ndarray
    lower_bound_func: Callable[[np.ndarray, float, bool], np.float64]
    weight_func: Callable[[dict, np.ndarray], np.ndarray]
    tuning_method: str
    weight_estimator: Optional[ClassifierMixin] = None
    alpha: Optional[np.ndarray] = None
    min_combination: int = 1
    scalar_skip: int = 1
    delta: float = 0.05

    def __post_init__(self) -> None:
        if self.tuning_method not in {"exact_scalar", "exact_combination", "greedy_combination"}:
            raise ValueError("tuning_method must be one of ['exact_scalar', 'exact_combination', 'greedy_combination']")

        if self.estimator.estimator_name not in MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR:
            raise ValueError(f"estimator_name must be one of {MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR.keys()}")

    def estimate_policy_value_with_tune(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        """Estimate the policy value of evaluation policy with hyperparameter tuning.

        Args:
            bandit_feedback: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.float64: estimated policy value.
        """

        if self.alpha is None:
            self.alpha = np.ones(bandit_feedback["len_list"])

        if self.tuning_method == "greedy_combination":
            estimated_policy_value = self._tune_combination_with_greedy_pruning(
                bandit_feedback=bandit_feedback, action_dist=action_dist
            )

        elif self.tuning_method == "exact_scalar":
            estimated_policy_value = self._tune_scalar_value(bandit_feedback=bandit_feedback, action_dist=action_dist)

        else:
            raise NotImplementedError(f"tuning_method={self.tuning_method} is not implemented yet.")

        return estimated_policy_value

    def _tune_combination_with_greedy_pruning(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        """Tune the hyperparameter with greedy pruning.

        Args:
            bandit_feedback: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.float64: estimated policy value.
        """

        theta_list, cnf_list, param_list = [], [], []
        current_param, C = self.hyper_param.copy(), np.sqrt(6) - 1
        bandit_feedback[self.param_name] = current_param

        # init
        kwargs = {self.param_name: current_param, "pi_a_x_e_estimator": self.weight_estimator}
        importance_weight = self.weight_func(
            data=bandit_feedback,
            action_dist=action_dist,
            behavior_assumption=self.estimator.behavior_assumption,
            **kwargs,
        )
        theta, cnf = self.estimator.estimate_policy_value_with_dev(
            reward=bandit_feedback["reward"],
            alpha=self.alpha,
            weight=importance_weight,
            lower_bound_func=self.lower_bound_func,
            delta=self.delta,
        )
        theta_list.append(theta), cnf_list.append(cnf)

        current_param_set = set(current_param)
        param_list.append(current_param_set)
        while len(current_param_set) > self.min_combination:
            theta_dict_, cnf_dict_, d_dict_, param_dict_ = {}, {}, {}, {}
            for i, d in enumerate(current_param_set):
                candidate_param = current_param_set.copy()
                candidate_param.remove(d)

                kwargs[self.param_name] = list(candidate_param)
                importance_weight = self.weight_func(
                    data=bandit_feedback,
                    action_dist=action_dist,
                    behavior_assumption=self.estimator.behavior_assumption,
                    **kwargs,
                )
                theta, cnf = self.estimator.estimate_policy_value_with_dev(
                    reward=bandit_feedback["reward"],
                    alpha=self.alpha,
                    weight=importance_weight,
                    lower_bound_func=self.lower_bound_func,
                    delta=self.delta,
                )

                d_dict_[i] = d
                theta_dict_[i] = theta
                cnf_dict_[i] = cnf
                param_dict_[i] = candidate_param.copy()

            idx_list = [i for i, _ in sorted(cnf_dict_.items(), key=lambda k: k[1], reverse=True)]
            for idx in idx_list:
                excluded_dim, param_i = d_dict_[idx], param_dict_[idx]
                theta_i, cnf_i = theta_dict_[idx], cnf_dict_[idx]
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                self.best_param = param_list[-1]
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                    param_list.append(param_i)
                else:
                    return theta_j[-1]

            current_param_set.remove(excluded_dim)

        return theta_j[-1]

    def _tune_combination_with_exact_pruning(self, bandit_feedback: dict, action_dist: np.ndarray):
        pass

    def _tune_scalar_value(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        """Tune the hyperparameter with scalar value.

        Args:
            bandit_feedback: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.float64: estimated policy value.
        """

        C = np.sqrt(6) - 1
        theta_dict_for_sort, cnf_dict_for_sort = {}, {}
        for id, unobs_cat_dim in enumerate(self.hyper_param):
            kwargs = {self.param_name: self.hyper_param[unobs_cat_dim:], "pi_a_x_e_estimator": self.weight_estimator}
            importance_weight = self.weight_func(
                data=bandit_feedback,
                action_dist=action_dist,
                behavior_assumption=self.estimator.behavior_assumption,
                **kwargs,
            )

            theta, cnf = self.estimator.estimate_policy_value_with_dev(
                reward=bandit_feedback["reward"],
                alpha=self.alpha,
                weight=importance_weight,
                lower_bound_func=self.lower_bound_func,
                delta=self.delta,
            )

            cnf_dict_for_sort[id] = cnf
            theta_dict_for_sort[id] = theta

        theta_list, cnf_list = [], []
        sorted_idx_list = [i for i, _ in sorted(cnf_dict_for_sort.items(), key=lambda k: k[1], reverse=True)]
        for i, idx in enumerate(sorted_idx_list):
            cnf_i = cnf_dict_for_sort[idx]
            theta_i = theta_dict_for_sort[idx]
            if len(theta_list) < 1:
                theta_list.append(theta_i), cnf_list.append(cnf_i)
            else:
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    best_idx = sorted_idx_list[i - 1]
                    return theta_dict_for_sort[best_idx]

        return theta_dict_for_sort[sorted_idx_list[-1]]


@dataclass
class Node:
    """Node class for User Behavior Tree.

    Args:
        node_id: int
            node id.

        user_behavior_id: int
            user behavior id.

        depth: int
            depth of the node.

        sample_id: np.ndarray
            sample (round) id.

        bootstrap_sample_ids: dict[int, np.ndarray]
            bootstrap sample ids.

    References:
        Kiyohara, Haruka, et al.
        "Off-policy evaluation of ranking policies under diverse user behavior.", 2023.

    """

    node_id: int
    user_behavior_id: int
    depth: int
    sample_id: np.ndarray
    bootstrap_sample_ids: dict[int, np.ndarray] = None


@dataclass
class UserBehaviorTree(BaseOffPolicyEstimatorWithTune):
    """User Behavior Tree class to estimate user behavior models for off-policy evaluation.

    Args:
        dataset: BaseBanditDataset
            dataset class for bandit feedback data.

        approximate_policy_value: float
            approximate policy value of the evaluation policy.

        weight_func: adaptive_weight
            function to compute the importance weight.

        candidate_weights: set[str]
            set of candidate user behavior names to optimize.

        eps: float
            parameter of the evaluation policy.

        len_list: int
            length of the recommendation list.

        n_partition: int
            number of partitions for the decision boundary.

        min_samples_leaf: int
            minimum number of samples in a leaf node.

        n_bootstrap: int
            number of bootstrap samples.

        noise_level: float
            noise level of true bias.

        alpha: Optional[np.ndarray]
            action choice matrix of shape (n_rounds, len_list).

        max_depth: Optional[int]
            maximum depth of the tree.

        random_state: Optional[int]
            random seed.

    References:
        Kiyohara, Haruka, et al.
        "Off-policy evaluation of ranking policies under diverse user behavior.", 2023.
    """

    dataset: BaseBanditDataset
    approximate_policy_value: float
    weight_func: adaptive_weight
    candidate_weights: set[str]
    eps: float
    len_list: int
    n_partition: int = 5
    min_samples_leaf: int = 10
    n_bootstrap: int = 10
    noise_level: float = 0.3
    alpha: Optional[np.ndarray] = None
    max_depth: Optional[int] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.estimator.estimator_name not in ADAPTIVE_ESTIMATORS_WITH_UBT_TO_BEHAVIOR:
            raise ValueError(f"estimator_name must be one of {ADAPTIVE_ESTIMATORS_WITH_UBT_TO_BEHAVIOR.keys()}")

        self.decision_boundary = dict()

        self.n_candidate_weights = len(self.candidate_weights)
        self.id2behavior = {i: c for i, c in enumerate(self.candidate_weights)}

        if self.max_depth is None:
            self.max_depth = np.infty

        if self.alpha is None:
            self.alpha = np.ones(self.len_list)

        self.random_ = check_random_state(self.random_state)

    def estimate_policy_value_with_tune(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        """Estimate the policy value of evaluation policy with hyperparameter tuning.

        Args:
            bandit_feedback: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.float64: estimated policy value.
        """

        estimated_user_behavior = self.fit_predict(data=bandit_feedback, action_dist=action_dist)
        kwargs_ = {self.param_name: estimated_user_behavior}
        importance_weight = self.weight_func(data=bandit_feedback, action_dist=action_dist, **kwargs_)

        estimated_value = self.estimator.estimate_policy_value(
            weight=importance_weight, alpha=self.alpha, reward=bandit_feedback["reward"]
        )

        return estimated_value

    def fit(self, data: dict, action_dist: np.ndarray) -> None:
        """Fit the User Behavior Tree model.

        Args:
            data: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy
        """

        self._create_input_dataset(data=data, action_dist=action_dist)

        # init
        node_queue = deque()
        base_user_behavior_id = self._select_base_user_behavior()
        node_id, depth = 0, 0
        sample_id = np.arange(self.train_size)
        bootstrap_sample_ids = {i: np.arange(self.train_size) for i in range(self.n_bootstrap)}

        initial_node = Node(
            node_id=node_id,
            sample_id=sample_id,
            user_behavior_id=base_user_behavior_id,
            depth=depth,
            bootstrap_sample_ids=bootstrap_sample_ids,
        )
        node_queue.append(initial_node)
        self._update_decision_boundary(node_id=node_id, user_behavior_id=base_user_behavior_id)
        self._update_global_pscore(node=initial_node)

        while len(node_queue):
            parent_node: Node = node_queue.pop()
            split_outcomes = self._search_split_boundary(parent_node)

            if split_outcomes["split_exist"]:
                # initialize child node
                left_node_id, right_node_id = node_id + 1, node_id + 2
                child_node_depth = parent_node.depth + 1
                left_node = Node(
                    node_id=left_node_id,
                    sample_id=split_outcomes["left_sample_id"],
                    user_behavior_id=split_outcomes["left_user_behavior_id"],
                    depth=child_node_depth,
                    bootstrap_sample_ids=split_outcomes["left_bootstrap_sample_ids"],
                )
                right_node = Node(
                    node_id=right_node_id,
                    sample_id=split_outcomes["right_sample_id"],
                    user_behavior_id=split_outcomes["right_user_behavior_id"],
                    depth=child_node_depth,
                    bootstrap_sample_ids=split_outcomes["right_bootstrap_sample_ids"],
                )
                self._update_decision_boundary(
                    node_id=left_node_id,
                    user_behavior_id=split_outcomes["left_user_behavior_id"],
                )
                self._update_decision_boundary(
                    node_id=right_node_id,
                    user_behavior_id=split_outcomes["right_user_behavior_id"],
                )
                self._update_global_pscore(left_node)
                self._update_global_pscore(right_node)

                # increment
                node_queue.append(left_node)
                node_queue.append(right_node)
                node_id += 2

    def fit_predict(self, data: dict, action_dist: np.ndarray) -> np.ndarray:
        """Fit the User Behavior Tree model and predict the estimated user behavior.

        Args:
            data: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.ndarray: array of estimated user behavior names.
        """

        self.fit(data=data, action_dist=action_dist)

        estimated_user_behavior = np.array([self.id2behavior[i] for i in self.train_reward_structure])
        return estimated_user_behavior

    def estimate_policy_value_with_best_param(self, bandit_feedback: dict, action_dist: np.ndarray) -> np.float64:
        self.test_n_samples = bandit_feedback["n_rounds"]
        self.test_context = bandit_feedback["context"]

        node_queue = deque()
        user_behavior_idx = np.zeros(self.test_n_samples, dtype=int)
        node_id, depth = 0, 0

        initial_node = Node(
            node_id=node_id,
            sample_id=np.arange(self.test_n_samples),
            user_behavior_id=self.decision_boundary[0]["parent_user_behavior_id"],
            depth=depth,
        )

        node_queue.append(initial_node)

        while len(node_queue):
            parent_node: Node = node_queue.pop()
            parent_node_id = parent_node.node_id
            parent_sample_id = parent_node.sample_id
            parent_depth = parent_node.depth
            parent_context = self.test_context[parent_sample_id]

            decision_boundary = self.decision_boundary[parent_node_id]
            split_exist = decision_boundary["split_exist"]

            if split_exist:
                boundary_feature = decision_boundary["feature_dim"]
                boundary_value = decision_boundary["feature_value"]

                left_sample_id = parent_sample_id[np.where(parent_context[:, boundary_feature] < boundary_value)]
                right_sample_id = parent_sample_id[np.where(parent_context[:, boundary_feature] >= boundary_value)]
                left_node = Node(
                    node_id=node_id + 1,
                    sample_id=left_sample_id,
                    user_behavior_id=self.decision_boundary[node_id + 1]["parent_user_behavior_id"],
                    depth=parent_depth + 1,
                )
                right_node = Node(
                    node_id=node_id + 2,
                    sample_id=right_sample_id,
                    user_behavior_id=self.decision_boundary[node_id + 2]["parent_user_behavior_id"],
                    depth=parent_depth + 1,
                )
                node_queue.append(left_node)
                node_queue.append(right_node)
                node_id += 2
            else:
                user_behavior_idx[parent_node.sample_id] = parent_node.user_behavior_id

        estimated_user_behavior = np.array([self.id2behavior[i] for i in user_behavior_idx])

        kwargs_ = {self.param_name: estimated_user_behavior}
        importance_weight = self.weight_func(data=bandit_feedback, action_dist=action_dist, **kwargs_)

        estimated_value = self.estimator.estimate_policy_value(
            weight=importance_weight, alpha=self.alpha, reward=bandit_feedback["reward"]
        )

        return estimated_value

    def _search_split_boundary(self, parent_node: Node) -> dict:
        """Search the best split boundary for the User Behavior Tree.

        Args:
            parent_node: Node
                parent node.

        Returns:
            dict: best split outcome. if outcome is exist, return left and right child nodes.
        """

        parent_node_id = parent_node.node_id
        parent_sample_id = parent_node.sample_id
        parent_bootstrap_sample_ids = parent_node.bootstrap_sample_ids
        parent_user_behavior_id = parent_node.user_behavior_id
        parent_depth = parent_node.depth
        parent_context = self.train_context[parent_sample_id]

        n_parent_samples = len(parent_sample_id)
        if n_parent_samples < 2 * self.min_samples_leaf:
            return {"split_exist": False}
        if parent_depth == self.max_depth:
            return {"split_exist": False}

        # init
        best_outcome = dict()
        best_outcome["split_exist"] = False
        best_mse = self._calc_mse_global()
        best_outcome["mse"] = best_mse

        split_feature_dims = self.random_.choice(self.train_context.shape[1], size=self.n_partition, replace=True)
        min_left_proportion = self.min_samples_leaf / n_parent_samples
        max_left_proportion = 1 - min_left_proportion
        split_left_proportions = self.random_.uniform(min_left_proportion, max_left_proportion, size=self.n_partition)
        feature_dim_sort_idx = np.argsort(split_feature_dims)
        split_feature_dims = split_feature_dims[feature_dim_sort_idx]
        split_left_proportions = split_left_proportions[feature_dim_sort_idx]

        for i in range(self.n_partition):
            feature_dim = split_feature_dims[i]
            if (i == 0) or (feature_dim != split_feature_dims[i - 1]):
                sorted_sample_id = parent_sample_id[np.argsort(parent_context[:, feature_dim])]

            split_id = int(split_left_proportions[i] * n_parent_samples)
            left_sample_id = sorted_sample_id[:split_id]
            right_sample_id = sorted_sample_id[split_id:]

            split_feature_value = (
                self.train_context[left_sample_id[-1], feature_dim]
                + self.train_context[right_sample_id[0], feature_dim]
            ) / 2

            (
                left_user_behavior_id,
                right_user_behavior_id,
                left_bootstrap_sample_ids,
                right_bootstrap_sample_ids,
                split_mse,
            ) = self._find_best_user_behavior(
                parent_sample_ids=parent_bootstrap_sample_ids,
                split_feature_dim=feature_dim,
                split_feature_value=split_feature_value,
            )
            if split_mse <= best_outcome["mse"]:
                best_outcome["split_exist"] = True
                best_outcome["mse"] = split_mse
                best_outcome["split_feature_dim"] = feature_dim
                best_outcome["split_feature_value"] = split_feature_value
                best_outcome["left_sample_id"] = left_sample_id
                best_outcome["right_sample_id"] = right_sample_id
                best_outcome["left_bootstrap_sample_ids"] = left_bootstrap_sample_ids
                best_outcome["right_bootstrap_sample_ids"] = right_bootstrap_sample_ids
                best_outcome["left_user_behavior_id"] = left_user_behavior_id
                best_outcome["right_user_behavior_id"] = right_user_behavior_id

        if best_outcome["split_exist"]:
            self._update_decision_boundary(
                node_id=parent_node_id,
                user_behavior_id=parent_user_behavior_id,
                split_exist=True,
                feature_dim=best_outcome["split_feature_dim"],
                feature_value=best_outcome["split_feature_value"],
            )

        return best_outcome

    def _find_best_user_behavior(
        self,
        parent_sample_ids: dict[int, np.ndarray],
        split_feature_dim: int,
        split_feature_value: float,
    ) -> tuple:
        """Find the best user behavior for the User Behavior Tree.

        Args:
            parent_sample_ids: dict[int, np.ndarray]
                parent sample ids.

            split_feature_dim: int
                split feature dimension.

            split_feature_value: float
                split feature value

        Returns:
            tuple: best user behavior for left and right child nodes.
        """

        estimated_values = np.zeros((self.n_candidate_weights, self.n_candidate_weights, self.n_bootstrap))
        left_sample_ids, right_sample_ids = {}, {}
        for i, bootstrap_data_ in enumerate(self.bootstrap_dataset):
            context, reward = bootstrap_data_["context"], bootstrap_data_["reward"]
            importance_weight = bootstrap_data_["importance_weight_dict"]
            weighted_reward_ = bootstrap_data_["weighted_reward"].copy()

            sample_id = parent_sample_ids[i].copy()
            context = context[sample_id]

            left_context_mask = np.where(context[:, split_feature_dim] < split_feature_value)
            right_context_mask = np.where(context[:, split_feature_dim] >= split_feature_value)
            left_sample_id_, right_sample_id_ = (
                sample_id[left_context_mask],
                sample_id[right_context_mask],
            )
            left_sample_ids[i] = left_sample_id_
            right_sample_ids[i] = right_sample_id_

            for left_user_behavior_id in range(self.n_candidate_weights):
                weighted_reward_[left_sample_id_] = self.estimator._estimate_round_rewards(
                    weight=importance_weight[left_user_behavior_id][left_sample_id_],
                    alpha=self.alpha,
                    reward=reward[left_sample_id_],
                )

                for right_user_behavior_id in range(self.n_candidate_weights):
                    weighted_reward_[right_sample_id_] = self.estimator._estimate_round_rewards(
                        weight=importance_weight[right_user_behavior_id][right_sample_id_],
                        alpha=self.alpha,
                        reward=reward[right_sample_id_],
                    )
                    estimated_values[left_user_behavior_id, right_user_behavior_id, i] = weighted_reward_.mean()

        surrogate_mse = self._compute_surrogate_mse(estimated_values, axis=2)
        best_left_user_behavior, best_right_user_behavior = np.unravel_index(
            surrogate_mse.argmin(), surrogate_mse.shape
        )
        split_outcome = (
            best_left_user_behavior,
            best_right_user_behavior,
            left_sample_ids,
            right_sample_ids,
            surrogate_mse.min(),
        )
        return split_outcome

    def _calc_mse_global(self) -> None:
        """Calculate the global mean squared error of the User Behavior Tree."""

        estimated_values = []
        for bootstrap_data_ in self.bootstrap_dataset:
            estimated_values.append(bootstrap_data_["weighted_reward"].mean())

        estimated_values = np.array(estimated_values)
        surrogate_mse = self._compute_surrogate_mse(estimated_values, axis=0)
        return surrogate_mse

    def _update_global_pscore(self, node: Node) -> None:
        """Update the global propensity score of the User Behavior Tree."""

        user_behavior_id = node.user_behavior_id
        sample_id = node.sample_id
        bootstrap_sample_ids = node.bootstrap_sample_ids

        self.train_reward_structure[sample_id] = user_behavior_id
        self.train_weighted_reward[sample_id] = self.estimator._estimate_round_rewards(
            weight=self.train_importance_weight_dict[user_behavior_id][sample_id],
            alpha=self.alpha,
            reward=self.train_reward[sample_id],
        )

        for i, bootstrap_data_ in enumerate(self.bootstrap_dataset):
            sample_id = bootstrap_sample_ids[i]
            iw_dict, reward = (
                bootstrap_data_["importance_weight_dict"],
                bootstrap_data_["reward"],
            )

            bootstrap_data_["reward_structure"][sample_id] = user_behavior_id
            bootstrap_data_["weighted_reward"][sample_id] = self.estimator._estimate_round_rewards(
                weight=iw_dict[user_behavior_id][sample_id],
                alpha=self.alpha,
                reward=reward[sample_id],
            )

    def _update_decision_boundary(
        self,
        node_id: int,
        user_behavior_id: int,
        split_exist: Optional[bool] = False,
        feature_dim: Optional[int] = None,
        feature_value: Optional[float] = None,
    ) -> None:
        """Update the decision boundary of the User Behavior Tree. if only `node_id` and `user_behavior_id` exists,
        Initialize the decision boundary.

        Args:
            node_id: int
                node id.

            user_behavior_id: int
                user behavior id.

            split_exist: Optional[bool] = False
                flag to determine whether the split boundary exists.

            feature_dim: Optional[int] = None
                feature dimension of the split boundary.

            feature_value: Optional[float] = None
                feature value of the split boundary.
        """

        self.decision_boundary[node_id] = {
            "parent_user_behavior_id": user_behavior_id,
            "split_exist": split_exist,
            "feature_dim": feature_dim,
            "feature_value": feature_value,
        }

    def _select_base_user_behavior(self) -> np.int64:
        """Select the base user behavior for the User Behavior Tree.

        Returns:
            np.int64: base user behavior id.
        """

        estimated_values = np.zeros((self.n_candidate_weights, self.n_bootstrap))
        for i, bootstrap_data_ in enumerate(self.bootstrap_dataset):
            reward = bootstrap_data_["reward"]
            importance_weight_dict = bootstrap_data_["importance_weight_dict"]

            for user_behavior_id, iw in importance_weight_dict.items():
                estimated_value = self.estimator.estimate_policy_value(weight=iw, alpha=self.alpha, reward=reward)
                estimated_values[user_behavior_id, i] = estimated_value

        surrogate_mse = self._compute_surrogate_mse(estimated_values=estimated_values, axis=1)
        best_user_behavior_id = surrogate_mse.argmin()
        return best_user_behavior_id

    def _compute_surrogate_mse(self, estimated_values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the surrogate mean squared error of the User Behavior Tree. make noise level of true bias.

        Args:
            estimated_values: np.ndarray
                estimated values of the User Behavior Tree.

            axis: int
                axis to compute the surrogate mean squared

        Returns:
            np.ndarray: surrogate mean squared error.
        """

        bias = estimated_values.mean(axis) - self.approximate_policy_value
        bias_hat = self.random_.normal(loc=bias, scale=self.noise_level * np.abs(bias))
        unbiased_variance = estimated_values.var(axis, ddof=1)
        surrogate_mse = (bias_hat**2) + unbiased_variance
        return surrogate_mse

    def _create_input_dataset(self, data: dict, action_dist: np.ndarray) -> None:
        """Create the input dataset for the User Behavior Tree.

        Args:
            data: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy
        """

        self.train_size = len(data["context"])
        self._create_bootstrap_dataset()

        self.train_importance_weight_dict = self._compute_initialized_importance_weight(
            data=data, action_dist=action_dist
        )
        self.train_context = data["context"]
        self.train_reward = data["reward"]
        self.train_reward_structure = np.zeros(self.train_size, dtype=int)
        self.train_weighted_reward = np.zeros(self.train_size)

    def _create_bootstrap_dataset(self) -> None:
        """Create the bootstrap dataset for the User Behavior Tree."""

        self.bootstrap_dataset = []
        for _ in range(self.n_bootstrap):
            bootstrap_data_ = self.dataset.obtain_batch_bandit_feedback(n_rounds=self.train_size)
            policy_logit_ = bootstrap_data_["evaluation_policy_logit"]
            action_dist_ = gen_eps_greedy(expected_reward=policy_logit_, eps=self.eps)

            importance_weight_dict_ = self._compute_initialized_importance_weight(
                data=bootstrap_data_, action_dist=action_dist_
            )

            self.bootstrap_dataset.append(
                dict(
                    context=bootstrap_data_["context"],
                    reward=bootstrap_data_["reward"],
                    importance_weight_dict=importance_weight_dict_,
                    reward_structure=np.zeros(self.train_size, dtype=int),
                    weighted_reward=np.zeros(self.train_size),
                )
            )

    def _compute_initialized_importance_weight(self, data: dict, action_dist: np.ndarray) -> dict:
        """Compute the initialized (all) importance weight for the User Behavior Tree.

        Args:
            data: dict
                bandit feedback data.

            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            dict: initialized importance weight for each candidate behaviors.
        """

        importance_weight_dict = dict()
        for user_behavior_id, behavior_name in self.id2behavior.items():
            weight_ = vanilla_weight(data=data, action_dist=action_dist, behavior_assumption=behavior_name)
            importance_weight_dict[user_behavior_id] = weight_

        return importance_weight_dict
