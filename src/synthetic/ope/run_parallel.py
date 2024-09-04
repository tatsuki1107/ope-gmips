import numpy as np
from obp.dataset import BaseBanditDataset

from ope import SlateOffPolicyEvaluation
from policy import gen_eps_greedy


def calc_avg_reward(args: tuple[BaseBanditDataset, int]) -> np.float64:
    dataset, test_size = args

    test_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_size, is_online=True)
    avg_reward = test_data["expected_reward_factual"].sum(1).mean()
    return avg_reward


def simulate_evaluation(args: tuple[list, BaseBanditDataset, int, float]) -> dict:
    ope_estimators, dataset, val_size, eps = args

    val_data = dataset.obtain_batch_bandit_feedback(n_rounds=val_size)

    # evaluation policy
    evaluation_policy = gen_eps_greedy(val_data["evaluation_policy_logit"], eps=eps)
    evaluation_pscore_dict = dataset.aggregate_propensity_score(
        pi_k=evaluation_policy,
        slate_id_at_k=val_data["slate_id_at_k"],
        p_e_d_a_k=val_data["p_e_d_a_k"],
        slate_action_context=val_data["action_context"],
    )

    # off policy evaluation
    ope = SlateOffPolicyEvaluation(bandit_feedback=val_data, ope_estimators=ope_estimators)
    estimated_policy_values = ope.estimate_policy_values(
        action_dist=evaluation_policy, evaluation_policy_pscore_dict=evaluation_pscore_dict
    )

    return estimated_policy_values
