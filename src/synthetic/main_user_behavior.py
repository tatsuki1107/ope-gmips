import logging
from multiprocessing import Pool
from multiprocessing import cpu_count
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from tqdm import tqdm

from dataset import SyntheticRankingDatasetWithActionEmbeds
from ope import MarginalizedIPSForRanking as MIPS
from ope import SelfNormalizedIPSForRanking as SNIPS
from ope import UserBehaviorTree
from ope import adaptive_weight
from ope import simulate_evaluation
from utils import TQDM_FORMAT
from utils import aggregate_simulation_results
from utils import visualize_mean_squared_error


cs = ConfigStore.instance()

logger = logging.getLogger(__name__)

ope_estimators = [
    MIPS(estimator_name="MSIPS"),
    MIPS(estimator_name="MIIPS"),
    MIPS(estimator_name="MRIPS"),
    SNIPS(estimator_name="AIPS (true)"),
]
# for the AIPS estimator with UserBehaviorTree
candidate_weights = {"independent", "cascade", "neighbor_3", "inverse_cascade"}


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
            n_actions_at_k=cfg.variation.n_unique_actions_at_k,
            dim_context=cfg.dim_context,
            n_cat_dim=cfg.variation.n_cat_dim,
            n_cat_per_dim=cfg.variation.n_cat_per_dim,
            n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
            len_list=cfg.variation.len_list,
            behavior_params=observed_behaviors,
            random_state=cfg.random_state,
            reward_noise=cfg.reward_noise,
            interaction_noise=cfg.interaction_noise,
            beta=cfg.beta,
            eps=cfg.eps,
        )
        ope_estimators_tune = [
            UserBehaviorTree(
                estimator=SNIPS(estimator_name=r"snAIPS-$\hat{c}$(UBT)"),
                param_name="estimated_user_behavior",
                bias_estimation_method="experimental_on_policy",
                dataset=dataset,
                weight_func=adaptive_weight,
                candidate_weights=candidate_weights,
                eps=cfg.eps,
                val_size=cfg.variation.val_size_bias_estimation,
                len_list=cfg.variation.len_list,
                n_partition=cfg.variation.n_partition,
                min_samples_leaf=cfg.variation.min_samples_leaf,
                n_bootstrap=cfg.variation.n_bootstrap,
                max_depth=cfg.variation.max_depth,
            )
        ]

        # calculate ground truth policy value (on policy)
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.test_size, is_online=True)
        policy_value = test_data["expected_reward_factual"].sum(1).mean()

        message = f"behavior_complexity={complexity}, included user behavior={list(observed_behaviors.keys())}..."
        args = (ope_estimators, ope_estimators_tune, dataset, cfg.val_size, cfg.eps)
        job_args = [args for _ in range(cfg.n_val_seeds)]
        with Pool(cpu_count() - 1) as pool:
            imap_iter = pool.imap(simulate_evaluation, job_args)
            tqdm_ = tqdm(imap_iter, total=cfg.n_val_seeds, desc=message, bar_format=TQDM_FORMAT)
            result_list = list(tqdm_)

        logger.info(tqdm_)
        # calculate MSE
        result_df = aggregate_simulation_results(
            simulation_result_list=result_list,
            policy_value=policy_value,
            x_value=complexity,
        )
        result_df_list.append(result_df)

    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(result_path / "result.csv")

    for yscale in ["linear", "log"]:
        for is_only_mse in [True, False]:
            img_path = result_path / f"{yscale}_mse_only={is_only_mse}_varying=behavior_complexity.png"
            visualize_mean_squared_error(
                result_df=result_df,
                xlabel="complexity of user behavior",
                img_path=img_path,
                yscale=yscale,
                xscale="linear",
                is_only_mse=is_only_mse,
            )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
