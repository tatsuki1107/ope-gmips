from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from obp.dataset import linear_behavior_policy
from obp.dataset import logistic_reward_function
from obp.utils import sample_action_fast
from obp.utils import softmax
from sklearn.utils import check_random_state
from utils.policy import gen_eps_greedy


class BaseBanditDataset(metaclass=ABCMeta):
    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        raise NotImplementedError


@dataclass
class SyntheticSlateDatasetWithActionEmbeds(BaseBanditDataset):
    n_actions: int
    dim_context: int
    n_category: int
    len_list: int
    behavior_ratio: dict
    is_category_probabialistic: bool = True
    p_e_a_param_std: float = 1.0
    reward_noise: float = 1.0
    interaction_noise: float = 0.5
    beta: float = 1.0
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

        # 未知分布の定義
        # p(e|x,a)
        self.p_e_a = softmax(self.random_.normal(scale=self.p_e_a_param_std, size=(self.n_actions, self.n_category)))

        self.interaction_params = self.random_.normal(scale=self.interaction_noise, size=(self.len_list,))

    def obtain_batch_bandit_feedback(self, n_rounds: int, is_online: bool = False) -> dict:
        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        action_context = np.eye(self.n_actions)
        policy_logits = linear_behavior_policy(
            context=context,
            action_context=action_context,
            random_state=self.random_state,
        )

        pi_b = gen_eps_greedy(policy_logits) if is_online else softmax(self.beta * policy_logits)

        slate_action, slate_category = [], []
        for _ in range(self.len_list):
            # a ~ \pi_{\cdot}(\cdot|x)
            # 各シュミレーションごとの周辺化操作を簡略化するため、復元抽出.
            action = sample_action_fast(pi_b)
            slate_action.append(action)

            # e ~ p(e|x,a)
            category = sample_action_fast(self.p_e_a[action])
            slate_category.append(category)

        slate_action, slate_category = (
            np.array(slate_action).T,
            np.array(slate_category).T,
        )

        # c ~ p(c|x)
        user_behavior = self.random_.choice(
            list(self.behavior_ratio.keys()),
            p=list(self.behavior_ratio.values()),
            size=n_rounds,
        )

        latent_cat_param = np.eye(self.n_category)
        q_x_e = logistic_reward_function(
            context=context,
            action_context=latent_cat_param,
            random_state=self.random_state,
        )

        # \mathbb{E}[r_i|x_i, a_i, e_i]
        expected_reward_factual = q_x_e[np.arange(n_rounds)[:, None], slate_category].copy()
        expected_reward_factual_fixed = action_interaction_reward_function(
            expected_reward_factual=expected_reward_factual,
            user_behavior=user_behavior,
            interaction_params=self.interaction_params,
        )

        # r_i ~ p(r_i|x_i, a_i, e_i)
        reward = self.random_.normal(expected_reward_factual_fixed, scale=self.reward_noise)

        pscore_dict = self.aggregate_propensity_score(
            pi=pi_b,
            slate_action=slate_action,
            p_e_a=self.p_e_a,
            slate_category=slate_category,
        )

        return dict(
            context=context,
            user_behavior=user_behavior,
            action=slate_action,
            category=slate_category,
            reward=reward,
            pscore=pscore_dict,
            evaluation_policy_logit=policy_logits,
            expected_reward_factual=expected_reward_factual_fixed,
            p_e_a=self.p_e_a,
        )

    def aggregate_propensity_score(
        self,
        pi: np.ndarray,
        slate_action: np.ndarray,
        p_e_a: np.ndarray,
        slate_category: np.ndarray,
    ) -> dict:
        rounds = np.arange(len(slate_action))
        # \pi__{\cdot}(\mathbf{a}_{i}(k)|x_i)
        pscore = pi[rounds[:, None], slate_action].copy()

        # p(e|x) = \sum_{a} p(e|x,a) p(a|x)
        pi_e = pi @ p_e_a

        # \pi_{\cdot}(\mathbf{e}_{i}(k)|x_i)
        pscore_e = pi_e[rounds[:, None], slate_category].copy()

        pscore_dict = dict(action=pscore, category=pscore_e)

        return pscore_dict


def action_interaction_reward_function(
    expected_reward_factual: np.ndarray,
    user_behavior: np.ndarray,
    interaction_params: np.ndarray,
) -> np.ndarray:
    len_list = len(interaction_params)
    expected_reward_factual_fixed = expected_reward_factual.copy()

    for behavior in np.unique(user_behavior):
        c = user_behavior == behavior

        if behavior == "independent":
            continue

        for pos_ in range(len_list):
            if behavior == "cascade":
                expected_reward_factual_fixed[c, pos_] += (
                    expected_reward_factual[c, :pos_] * interaction_params[:pos_]
                ).sum(axis=1)

            elif behavior == "all":
                expected_reward_factual_fixed[c, pos_] += (
                    expected_reward_factual[c, :pos_] * interaction_params[:pos_]
                ).sum(axis=1)

                expected_reward_factual_fixed[c, pos_] += (
                    expected_reward_factual[c, pos_ + 1 :] * interaction_params[pos_ + 1 :]
                ).sum(axis=1)

    return expected_reward_factual_fixed
