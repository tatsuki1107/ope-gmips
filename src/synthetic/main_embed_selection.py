import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import SyntheticRankingDatasetWithActionEmbeds
from ope import EmbeddingSelectionWithSLOPE
from ope import MarginalizedIPSForRanking as MIPS
from ope import RankingOffPolicyEvaluation
from ope import RankingOffPolicyEvaluationWithTune
from ope.importance_weight import marginalized_weight
from policy import gen_eps_greedy
from utils import TQDM_FORMAT
from utils import aggregate_simulation_results
from utils import estimate_student_t_lower_bound
from utils import visualize_mean_squared_error


cs = ConfigStore.instance()

logger = logging.getLogger(__name__)

ope_estimators = [
    MIPS(estimator_name="MSIPS"),
    MIPS(estimator_name="MRIPS"),
]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start the experiment. the settings are as follows")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)
    result_path = log_path
    result_path.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticRankingDatasetWithActionEmbeds(
        n_actions_at_k=cfg.n_unique_actions_at_k,
        dim_context=cfg.dim_context,
        n_cat_dim=cfg.variation.n_cat_dim,
        n_cat_per_dim=cfg.variation.n_cat_per_dim,
        n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
        n_deficient_actions_at_k=cfg.n_deficient_actions_at_k,
        len_list=cfg.len_list,
        behavior_params=cfg.variation.user_behaviors,
        random_state=cfg.random_state,
        reward_noise=cfg.reward_noise,
        interaction_noise=cfg.interaction_noise,
        beta=cfg.beta,
        eps=cfg.eps,
    )

    # calculate ground truth policy value (on policy)
    test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.test_size, is_online=True)
    policy_value = test_data["expected_reward_factual"].sum(1).mean()

    result_df_list = []
    for val_size in cfg.variation.val_size_list:
        message = f"val_size={val_size}"
        tqdm_ = tqdm(range(cfg.n_val_seeds), desc=message, bar_format=TQDM_FORMAT)
        result_list = []
        for seed in tqdm_:
            # generate synthetic data
            val_data = dataset.obtain_batch_bandit_feedback(n_rounds=val_size)

            # evaluation policy
            action_dist = gen_eps_greedy(expected_reward=val_data["evaluation_policy_logit"], eps=cfg.eps)

            if seed % 100 == 0:
                # define tuning estimators before ope
                ope_estimators_tune = [
                    EmbeddingSelectionWithSLOPE(
                        estimator=MIPS(estimator_name="MSIPS (w/SLOPE)"),
                        param_name="action_embed_dim",
                        hyper_param=np.arange(cfg.variation.n_cat_dim),
                        lower_bound_func=estimate_student_t_lower_bound,
                        weight_func=marginalized_weight,
                        tuning_method="exact_scalar",
                    ),
                    EmbeddingSelectionWithSLOPE(
                        estimator=MIPS(estimator_name="MRIPS (w/SLOPE)"),
                        param_name="action_embed_dim",
                        hyper_param=np.arange(cfg.variation.n_cat_dim),
                        lower_bound_func=estimate_student_t_lower_bound,
                        weight_func=marginalized_weight,
                        tuning_method="exact_scalar",
                    ),
                ]
                ope_tune = RankingOffPolicyEvaluationWithTune(
                    ope_estimators_tune=ope_estimators_tune,
                )
                estimated_policy_values_with_tune = ope_tune.estimate_policy_values_with_tune(
                    bandit_feedback=val_data.copy(), action_dist=action_dist
                )
            else:
                estimated_policy_values_with_tune = ope_tune.estimate_policy_values_with_best_param(
                    bandit_feedback=val_data.copy(), action_dist=action_dist
                )

            # off policy evaluation
            ope = RankingOffPolicyEvaluation(
                bandit_feedback=val_data,
                ope_estimators=ope_estimators,
            )
            estimated_policy_values = ope.estimate_policy_values(action_dist=action_dist)
            estimated_policy_values = estimated_policy_values | estimated_policy_values_with_tune
            result_list.append(estimated_policy_values)

        logger.info(tqdm_)
        # calculate MSE
        result_df = aggregate_simulation_results(
            simulation_result_list=result_list, policy_value=policy_value, x_value=val_size
        )
        result_df_list.append(result_df)

    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(result_path / "result.csv")

    visualize_mean_squared_error(
        result_df=result_df,
        xlabel="sample sizes of logged data",
        img_path=result_path,
        xscale="log",
    )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
