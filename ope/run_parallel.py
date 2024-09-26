from obp.dataset import BaseBanditDataset

from ope import RankingOffPolicyEvaluation
from policy import gen_eps_greedy


def simulate_evaluation(args: tuple[list, list, BaseBanditDataset, int, float]) -> dict:
    ope_estimators, ope_estimators_tune, dataset, val_size, eps = args

    val_data = dataset.obtain_batch_bandit_feedback(n_rounds=val_size)

    # evaluation policy
    action_dist = gen_eps_greedy(val_data["evaluation_policy_logit"], eps=eps)

    # off policy evaluation
    ope = RankingOffPolicyEvaluation(
        bandit_feedback=val_data, ope_estimators=ope_estimators, ope_estimators_tune=ope_estimators_tune
    )
    estimated_policy_values = ope.estimate_policy_values(action_dist=action_dist)

    return estimated_policy_values
