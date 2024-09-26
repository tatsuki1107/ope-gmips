import logging
from multiprocessing import Pool
from multiprocessing import cpu_count
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import SyntheticRankingDatasetWithActionEmbeds
from ope import SLOPE
from ope import MarginalizedIPSForRanking as MIPS
from ope import simulate_evaluation
from ope.importance_weight import marginalized_weight
from utils import TQDM_FORMAT
from utils import aggregate_simulation_results
from utils import estimate_student_t_lower_bound
from utils import visualize_mean_squared_error


cs = ConfigStore.instance()

logger = logging.getLogger(__name__)

ope_estimators = {
    "stanadrd": [MIPS(estimator_name="MSIPS")],
    "independent": [MIPS(estimator_name="MIIPS")],
    "cascade": [MIPS(estimator_name="MRIPS")],
}
ope_estimator_with_slope = {
    "stanadrd": MIPS(estimator_name="MSIPS (SLOPE)"),
    "independent": MIPS(estimator_name="MIIPS (SLOPE)"),
    "cascade": MIPS(estimator_name="MRIPS (SLOPE)"),
}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start the experiment. the settings are as follows")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)

    user_behavior = cfg.variation.user_behavior
    result_path = log_path / user_behavior
    result_path.mkdir(parents=True, exist_ok=True)

    behavior_params = {user_behavior: 1.0}
    dataset = SyntheticRankingDatasetWithActionEmbeds(
        n_actions_at_k=cfg.n_unique_actions_at_k,
        dim_context=cfg.dim_context,
        n_cat_dim=cfg.n_cat_dim,
        n_cat_per_dim=cfg.n_cat_per_dim,
        n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
        len_list=cfg.len_list,
        behavior_params=behavior_params,
        random_state=cfg.random_state,
        reward_noise=cfg.reward_noise,
        interaction_noise=cfg.interaction_noise,
        beta=cfg.beta,
        eps=cfg.eps,
    )
    ope_estimators_tune = [
        SLOPE(
            estimator=ope_estimator_with_slope[user_behavior],
            param_name="action_embed_dim",
            hyper_param=np.arange(cfg.n_cat_dim),
            lower_bound_func=estimate_student_t_lower_bound,
            tuning_method="greedy_combination",
            weight_func=marginalized_weight,
            min_combination=cfg.variation.min_combination,
            delta=cfg.variation.delta,
        )
    ]

    result_df_list = []
    for val_size in cfg.variation.val_size_list:
        # calculate ground truth policy value (on policy)
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.test_size, is_online=True)
        policy_value = test_data["expected_reward_factual"].sum(1).mean()

        message = f"behavior={user_behavior}, val_size={val_size}"
        args = (ope_estimators[user_behavior], ope_estimators_tune, dataset, cfg.val_size, cfg.eps)
        job_args = [args for _ in range(cfg.n_val_seeds)]
        with Pool(cpu_count() - 1) as pool:
            imap_iter = pool.imap(simulate_evaluation, job_args)
            tqdm_ = tqdm(imap_iter, total=cfg.n_val_seeds, desc=message, bar_format=TQDM_FORMAT)
            result_list = list(tqdm_)

        logger.info(tqdm_)
        # calculate MSE
        result_df = aggregate_simulation_results(
            simulation_result_list=result_list, policy_value=policy_value, x_value=val_size
        )
        result_df_list.append(result_df)

    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(result_path / "result.csv")

    for yscale in ["linear", "log"]:
        for is_only_mse in [True, False]:
            img_path = result_path / f"{yscale}_mse_only={is_only_mse}_varying=val_size_{user_behavior}.png"
            visualize_mean_squared_error(
                result_df=result_df,
                xlabel="sample sizes",
                img_path=img_path,
                yscale=yscale,
                xscale="log",
                is_only_mse=is_only_mse,
            )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
