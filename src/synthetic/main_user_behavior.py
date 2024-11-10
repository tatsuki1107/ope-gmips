import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from tqdm import tqdm

from dataset import SyntheticRankingDatasetWithActionEmbeds
from ope import MarginalizedIPSForRanking as MIPS
from ope import RankingOffPolicyEvaluation
from ope import SelfNormalizedIPSForRanking as SNIPS
from ope import UserBehaviorTree
from ope.importance_weight import adaptive_weight
from ope.meta_ranking import RankingOffPolicyEvaluationWithTune
from policy import gen_eps_greedy
from utils import TQDM_FORMAT
from utils import aggregate_simulation_results
from utils import visualize_mean_squared_error


cs = ConfigStore.instance()

logger = logging.getLogger(__name__)

ope_estimators = [
    MIPS(estimator_name="MSIPS"),
    MIPS(estimator_name="MIIPS"),
    SNIPS(estimator_name="snRIPS"),
    MIPS(estimator_name="MRIPS"),
]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start the experiment. the settings are as follows")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)

    result_path = log_path
    result_path.mkdir(parents=True, exist_ok=True)

    observed_behaviors = dict()
    result_df_list = []
    for behavior_name, complexity in cfg.variation.behavior_complexity_dict.items():
        observed_behaviors[behavior_name] = 1.0

        dataset = SyntheticRankingDatasetWithActionEmbeds(
            n_actions_at_k=cfg.n_unique_actions_at_k,
            dim_context=cfg.dim_context,
            n_cat_dim=cfg.n_cat_dim,
            n_cat_per_dim=cfg.n_cat_per_dim,
            n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
            n_deficient_actions_at_k=cfg.n_deficient_actions_at_k,
            len_list=cfg.len_list,
            behavior_params=observed_behaviors,
            random_state=cfg.random_state,
            reward_noise=cfg.reward_noise,
            interaction_noise=cfg.interaction_noise,
            beta=cfg.beta,
            eps=cfg.eps,
        )
        # calculate ground truth policy value (on policy)
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.test_size, is_online=True)
        position_wise_policy_values = test_data["expected_reward_factual"].mean(0)
        policy_value = position_wise_policy_values.sum()

        message = f"behavior_complexity={complexity}, included user behavior={list(observed_behaviors.keys())}"
        tqdm_ = tqdm(range(cfg.n_val_seeds), desc=message, bar_format=TQDM_FORMAT)
        result_list = []
        for seed in tqdm_:
            # generate synthetic data
            val_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.val_size)

            # evaluation policy
            action_dist = gen_eps_greedy(expected_reward=val_data["evaluation_policy_logit"], eps=cfg.eps)

            if seed % 100 == 0:
                # define tuning estimators before ope
                ope_estimators_tune = [
                    UserBehaviorTree(
                        position_wise_policy_values=position_wise_policy_values,
                        estimator=SNIPS(estimator_name="snAIPS (w/UBT)"),
                        param_name="estimated_user_behavior",
                        dataset=dataset,
                        weight_func=adaptive_weight,
                        candidate_weights=cfg.variation.candidate_weights,
                        eps=cfg.eps,
                        len_list=cfg.len_list,
                        n_partition=cfg.variation.n_partition,
                        min_samples_leaf=cfg.variation.min_samples_leaf,
                        n_bootstrap=cfg.variation.n_bootstrap,
                        max_depth=cfg.variation.max_depth,
                        noise_level=cfg.variation.noise_level,
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
            simulation_result_list=result_list, policy_value=policy_value, x_value=complexity
        )
        result_df_list.append(result_df)

    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(result_path / "result.csv")

    visualize_mean_squared_error(
        result_df=result_df,
        xlabel="complexity of user behavior",
        img_path=result_path,
        xscale="linear",
    )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
