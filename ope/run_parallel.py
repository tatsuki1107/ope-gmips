from obp.dataset import BaseBanditDataset

from ope import RankingOffPolicyEvaluation
from policy import gen_eps_greedy


def simulate_evaluation(args: tuple[list, BaseBanditDataset, int, float]) -> dict:
    """simulate evaluation of off-policy evaluation for ranking to parallelize the evaluation process.

    Args:
        args: tuple[list, list, BaseBanditDataset, int, float]
            tuple of arguments to simulate evaluation of off-policy evaluation for ranking

    Returns:
        dict: estimated policy values of evaluation policy for each seed.
    """

    ope_estimators, dataset, val_size, eps = args

    val_data = dataset.obtain_batch_bandit_feedback(n_rounds=val_size)

    # evaluation policy
    action_dist = gen_eps_greedy(val_data["evaluation_policy_logit"], eps=eps)

    # off policy evaluation
    ope = RankingOffPolicyEvaluation(
        bandit_feedback=val_data,
        ope_estimators=ope_estimators,
    )
    estimated_policy_values = ope.estimate_policy_values(action_dist=action_dist)

    return estimated_policy_values
