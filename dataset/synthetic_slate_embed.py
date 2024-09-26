from dataclasses import dataclass

import numpy as np
from obp.dataset import BaseBanditDataset
from obp.dataset import logistic_reward_function
from obp.utils import sample_action_fast
from obp.utils import softmax
from sklearn.utils import check_random_state

from dataset.user_behavior import action_interaction_reward_function
from dataset.user_behavior import create_interaction_params
from dataset.user_behavior import linear_user_behavior_model
from policy import gen_eps_greedy
from utils import sample_slate_fast_with_replacement


@dataclass
class SyntheticRankingDatasetWithActionEmbeds(BaseBanditDataset):
    n_actions_at_k: int
    dim_context: int
    n_cat_dim: int
    n_cat_per_dim: int
    len_list: int
    behavior_params: dict[str, float]
    n_unobserved_cat_dim: int
    reward_noise: float
    interaction_noise: float
    beta: float
    eps: float
    random_state: int
    latent_param_mat_dim: int = 5
    p_e_a_param_std: float = 1.0
    delta: float = 1.0

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

        self.n_actions = self.n_actions_at_k * self.len_list

        # set of rankings
        self.candidate_action_set_at_k = np.arange(self.n_actions).reshape(self.len_list, self.n_actions_at_k)

        self.observed_cat_dim = np.arange(self.n_cat_dim)[self.n_unobserved_cat_dim :]

        # define the distribution of action embeddings
        self._define_action_embed()
        self.interaction_params = self.random_.uniform(0.0, self.interaction_noise, size=(self.len_list, self.len_list))

        # define the unknown distribution of the user behavior
        self.interaction_params = create_interaction_params(
            behavior_names=list(self.behavior_params.keys()),
            len_list=self.len_list,
            interaction_noise=self.interaction_noise,
            random_state=self.random_state,
        )

        self.gamma_z = np.array(list(self.behavior_params.values())) if len(self.behavior_params) > 1 else None
        self.id2user_behavior = {i: c for i, (c, _) in enumerate(self.behavior_params.items())}

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

        # c ~ p(c|x)
        p_c_x = linear_user_behavior_model(
            context=context,
            gamma_z=self.gamma_z,
            delta=self.delta,
            random_state=self.random_state,
        )
        user_behavior_id = sample_action_fast(p_c_x)
        user_behavior = np.array([self.id2user_behavior[i] for i in user_behavior_id])

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

        q_x_a_k = q_x_a[:, self.candidate_action_set_at_k.T]

        # a(k) ~ \pi_b(\cdot|x)
        pi_b = gen_eps_greedy(q_x_a_k, eps=self.eps) if is_online else softmax(self.beta * q_x_a_k)
        action_id_at_k, rankings = sample_slate_fast_with_replacement(
            pi_b, candidate_action_set_at_k=self.candidate_action_set_at_k
        )
        rounds = np.arange(n_rounds)[:, None]
        pscores = pi_b[rounds, action_id_at_k, np.arange(self.len_list)[None, :]]

        # e ~ p(e|x,a)
        slate_action_1d = rankings.reshape(-1)
        slate_embeddings = sample_slate_fast_with_replacement(self.p_e_d_a[slate_action_1d])
        slate_embeddings = slate_embeddings.reshape(n_rounds, self.len_list, self.n_cat_dim)

        cat_dim_importance_ = self.cat_dim_importance.reshape((1, self.n_cat_dim, 1))
        base_expected_reward_factual = []
        for pos_ in range(self.len_list):
            expected_reward_pos_ = (cat_dim_importance_ * q_x_e)[
                rounds, np.arange(self.n_cat_dim), slate_embeddings[:, pos_, :]
            ].sum(1)
            base_expected_reward_factual.append(expected_reward_pos_)

        base_expected_reward_factual = np.array(base_expected_reward_factual).T  # shape: (n_rounds, len_list)

        expected_reward_factual_fixed = action_interaction_reward_function(
            base_expected_reward_factual=base_expected_reward_factual,
            user_behavior=user_behavior,
            interaction_params=self.interaction_params,
        )

        # r_i ~ p(r_i|x_i, a_i, e_i)
        reward = self.random_.normal(expected_reward_factual_fixed, scale=self.reward_noise)

        p_e_d_a_k = self.p_e_d_a.reshape(self.len_list, self.n_actions_at_k, self.n_cat_per_dim, self.n_cat_dim)

        slate_embeddings = slate_embeddings[:, :, self.n_unobserved_cat_dim :]

        return dict(
            n_rounds=n_rounds,
            len_list=self.len_list,
            n_actions_at_k=self.n_actions_at_k,
            observed_cat_dim=self.observed_cat_dim,
            context=context,
            user_behavior=user_behavior,
            action=rankings,
            p_e_d_a_k=p_e_d_a_k,
            action_context=slate_embeddings,
            reward=reward,
            pscore=pscores,
            evaluation_policy_logit=q_x_a_k,
            expected_reward_factual=expected_reward_factual_fixed,
            action_id_at_k=action_id_at_k,
            pi_b=pi_b,
        )
