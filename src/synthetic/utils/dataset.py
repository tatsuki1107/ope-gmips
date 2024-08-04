from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from obp.dataset import logistic_reward_function
from obp.utils import softmax
from sklearn.utils import check_random_state
from utils.policy import gen_eps_greedy
from utils.sampling import sample_slate_fast_with_replacement


class BaseBanditDataset(metaclass=ABCMeta):
    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        raise NotImplementedError


@dataclass
class SyntheticSlateDatasetWithActionEmbeds(BaseBanditDataset):
    n_actions_at_k: int
    dim_context: int
    n_cat_dim: int
    n_cat_per_dim: int
    len_list: int
    behavior_ratio: dict
    n_unobserved_cat_dim: int
    reward_noise: float
    interaction_noise: float
    beta: float
    eps: float
    random_state: int
    latent_param_mat_dim: int = 5
    p_e_a_param_std: float = 1.0

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

        self.n_actions = self.n_actions_at_k * self.len_list

        # set of slates
        self.candidate_action_set_at_k = np.arange(self.n_actions).reshape(self.len_list, self.n_actions_at_k)

        # define the unknown distribution
        self._define_action_embed()
        self.interaction_params = self.random_.uniform(0.0, self.interaction_noise, size=(self.len_list, self.len_list))

    def _define_action_embed(self) -> None:
        # p(e_d|x,a)
        self.p_e_d_a = softmax(
            self.random_.normal(scale=self.p_e_a_param_std, size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim))
        )

        self.latent_cat_param = self.random_.normal(
            size=(self.n_cat_dim, self.n_cat_per_dim, self.latent_param_mat_dim)
        )

        self.cat_dim_importance = self.random_.dirichlet(
            alpha=self.random_.uniform(size=self.n_cat_dim),
            size=1,
        )

    def obtain_batch_bandit_feedback(self, n_rounds: int, is_online: bool = False) -> dict:
        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        # r
        q_x_a, q_x_e = [], []
        for d in range(self.n_cat_dim):
            q_x_e_d = logistic_reward_function(
                context=context,
                action_context=self.latent_cat_param[d],
                random_state=self.random_state + d,
            )
            q_x_a_d = q_x_e_d @ self.p_e_d_a[:, :, d].T

            q_x_a.append(q_x_a_d)
            q_x_e.append(q_x_e_d)

        q_x_a = np.array(q_x_a).transpose(1, 2, 0)  # shape: (n_rounds, n_actions, n_cat_dim)
        q_x_e = np.array(q_x_e).transpose(1, 0, 2)  # shape: (n_rounds, n_cat_dim, n_cat_per_dim)

        cat_dim_importance_ = self.cat_dim_importance.reshape((1, 1, self.n_cat_dim))
        q_x_a = (q_x_a * cat_dim_importance_).sum(2)  # shape: (n_rounds, n_actions)

        q_x_a_k = np.tile(q_x_a[:, :, None], reps=self.len_list)
        q_x_a_k = q_x_a_k[
            np.arange(n_rounds)[:, None, None],
            self.candidate_action_set_at_k.T[None, :, :],
            np.arange(self.len_list)[None, None, :],
        ]

        # a(k) ~ \pi_b(\cdot|x)
        pi_b_k = gen_eps_greedy(q_x_a_k, eps=self.eps) if is_online else softmax(self.beta * q_x_a_k)
        slate_id_at_k, slates = sample_slate_fast_with_replacement(
            pi_b_k, candidate_action_set_at_k=self.candidate_action_set_at_k
        )

        # e ~ p(e|x,a)
        slate_action_1d = slates.reshape(-1)
        slate_embeddings = sample_slate_fast_with_replacement(self.p_e_d_a[slate_action_1d])
        slate_embeddings = slate_embeddings.reshape(n_rounds, self.len_list, self.n_cat_dim)

        # c ~ p(c|x)
        user_behavior = self.random_.choice(
            list(self.behavior_ratio.keys()),
            p=list(self.behavior_ratio.values()),
            size=n_rounds,
        )

        cat_dim_importance_ = self.cat_dim_importance.reshape((1, self.n_cat_dim, 1))
        expected_reward_factual = []
        for pos_ in range(self.len_list):
            expected_reward_pos_ = (cat_dim_importance_ * q_x_e)[
                np.arange(n_rounds)[:, None], np.arange(self.n_cat_dim), slate_embeddings[:, pos_, :]
            ].sum(1)
            expected_reward_factual.append(expected_reward_pos_)

        expected_reward_factual = np.array(expected_reward_factual).T  # shape: (n_rounds, len_list)

        expected_reward_factual_fixed = action_interaction_reward_function(
            expected_reward_factual=expected_reward_factual,
            user_behavior=user_behavior,
            interaction_params=self.interaction_params,
        )

        # r_i ~ p(r_i|x_i, a_i, e_i)
        reward = self.random_.normal(expected_reward_factual_fixed, scale=self.reward_noise)

        p_e_d_a_k = self.p_e_d_a.reshape(self.n_actions_at_k, self.len_list, self.n_cat_per_dim, self.n_cat_dim)

        pscore_dict = self.aggregate_propensity_score(
            pi_k=pi_b_k,
            slate_id_at_k=slate_id_at_k,
            p_e_d_a_k=p_e_d_a_k[:, :, :, self.n_unobserved_cat_dim :],
            slate_action_context=slate_embeddings,
        )

        return dict(
            context=context,
            user_behavior=user_behavior,
            action=slates,
            p_e_d_a_k=p_e_d_a_k[:, :, :, self.n_unobserved_cat_dim :],
            action_context=slate_embeddings[:, :, self.n_unobserved_cat_dim :],
            reward=reward,
            pscore=pscore_dict,
            evaluation_policy_logit=q_x_a_k,
            expected_reward_factual=expected_reward_factual_fixed,
            slate_id_at_k=slate_id_at_k,
        )

    def aggregate_propensity_score(
        self,
        pi_k: np.ndarray,
        slate_id_at_k: np.ndarray,
        p_e_d_a_k: np.ndarray,
        slate_action_context: np.ndarray,
    ) -> dict:
        rounds = np.arange(len(slate_id_at_k))
        position = np.arange(self.len_list)
        marginal_pscore = np.ones_like(slate_id_at_k, dtype=float)
        for pos_ in position:
            for d in range(self.n_cat_dim):
                p_e_pi_d_k = pi_k[:, :, pos_] @ p_e_d_a_k[:, pos_, :, d]
                marginal_pscore[:, pos_] *= p_e_pi_d_k[rounds, slate_action_context[:, pos_, d]]

        # \pi__{\cdot}(\mathbf{a}_{i}(k)|x_i)
        pscore = pi_k[rounds[:, None], slate_id_at_k, position[None, :]].copy()

        pscore_dict = dict(action=pscore, category=marginal_pscore)

        return pscore_dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def action_interaction_reward_function(
    expected_reward_factual: np.ndarray,
    user_behavior: np.ndarray,
    interaction_params: np.ndarray,
) -> np.ndarray:
    position = np.arange(len(interaction_params))
    len_list = len(interaction_params)

    expected_reward_factual_fixed = expected_reward_factual.copy()
    expected_reward_factual_fixed /= len_list

    for behavior in np.unique(user_behavior):
        c = user_behavior == behavior

        if behavior == "independent":
            continue

        for pos_ in range(len_list):
            if behavior == "cascade":
                expected_reward_factual_fixed[c, pos_] += (
                    interaction_params[pos_, :pos_]
                    * expected_reward_factual_fixed[c, :pos_]
                    / np.abs(position[:pos_] - pos_)
                ).sum(axis=1)
            elif behavior == "standard":
                expected_reward_factual_fixed[c, pos_] += (
                    interaction_params[pos_, :pos_]
                    * expected_reward_factual_fixed[c, :pos_]
                    / np.abs(position[:pos_] - pos_)
                ).sum(axis=1)
                expected_reward_factual_fixed[c, pos_] += (
                    interaction_params[pos_, pos_ + 1 :]
                    * expected_reward_factual_fixed[c, pos_ + 1 :]
                    / np.abs(position[pos_ + 1 :] - pos_)
                ).sum(axis=1)
            else:
                raise NotImplementedError

    return expected_reward_factual_fixed
