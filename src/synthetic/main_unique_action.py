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

from dataset import SyntheticSlateDatasetWithActionEmbeds
from ope import IndependentIPS as IIPS
from ope import RewardInteractionIPS as RIPS
from ope import SelfNormalizedIndependentIPS as SNIIPS
from ope import SelfNormalizedRewardInteractionIPS as SNRIPS
from ope import SelfNormalizedStandardIPS as SNSIPS
from ope import StandardIPS as SIPS
from ope import calc_avg_reward
from ope import simulate_evaluation
from utils import aggregate_simulation_results
from utils import visualize_mean_squared_error


TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


cs = ConfigStore.instance()
# cs.store(name="setting", node=Config)

logger = logging.getLogger(__name__)

ope_estimators = [
    SNSIPS(estimator_name="snSIPS", pscore_type="action"),
    SIPS(estimator_name="MSIPS (true)", pscore_type="category"),
    SNIIPS(estimator_name="snIIPS", pscore_type="action"),
    IIPS(estimator_name="MIIPS (true)", pscore_type="category"),
    SNRIPS(estimator_name="snRIPS", pscore_type="action"),
    RIPS(estimator_name="MRIPS (true)", pscore_type="category"),
]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start experiment. the configure is as follow")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)

    for user_behavior in cfg.user_behavior:
        result_path = log_path / user_behavior
        result_path.mkdir(parents=True, exist_ok=True)

        behavior_ratio = {behavior: 1.0 if behavior == user_behavior else 0.0 for behavior in cfg.user_behavior}
        result_df_list = []
        for n_actions_at_k in cfg.variation.unique_action_at_k_list:
            dataset = SyntheticSlateDatasetWithActionEmbeds(
                n_actions_at_k=n_actions_at_k,
                dim_context=cfg.dim_context,
                n_cat_dim=cfg.variation.n_cat_dim,
                n_cat_per_dim=cfg.variation.n_cat_per_dim,
                n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
                len_list=cfg.variation.len_list,
                behavior_ratio=behavior_ratio,
                random_state=cfg.random_state,
                reward_noise=cfg.reward_noise,
                interaction_noise=cfg.interaction_noise,
                beta=cfg.beta,
                eps=cfg.eps,
            )

            job_args = [(dataset, cfg.test_size) for _ in range(cfg.n_test_seeds)]
            num_workers = cpu_count() - 1
            with Pool(num_workers) as pool:
                results = pool.imap(calc_avg_reward, job_args)
                tqdm_ = tqdm(
                    results,
                    total=cfg.n_test_seeds,
                    desc="calculate approximate policy value ...",
                    bar_format=TQDM_FORMAT,
                )
                approximate_policy_value = np.mean(list(tqdm_))

            logger.info(tqdm_)

            message = f"behavior={user_behavior}, n_unique_actions={dataset.n_actions}"
            job_args = [(ope_estimators, dataset, cfg.val_size, cfg.eps) for _ in range(cfg.n_val_seeds)]
            with Pool(num_workers) as pool:
                imap_iter = pool.imap(simulate_evaluation, job_args)
                tqdm_ = tqdm(imap_iter, total=cfg.n_val_seeds, desc=message, bar_format=TQDM_FORMAT)
                result_list = list(tqdm_)

            logger.info(tqdm_)
            # calculate MSE
            result_df = aggregate_simulation_results(
                simulation_result_list=result_list, policy_value=approximate_policy_value, x_value=dataset.n_actions
            )
            result_df_list.append(result_df)

        result_df = pd.concat(result_df_list).reset_index(level=0)
        result_df.to_csv(result_path / "result.csv")

        for yscale in ["linear", "log"]:
            for is_only_mse in [True, False]:
                img_path = result_path / f"{yscale}_mse_only={is_only_mse}_varying=unique_action_{user_behavior}.png"
                visualize_mean_squared_error(
                    result_df=result_df,
                    xlabel="the number of unique actions",
                    img_path=img_path,
                    yscale=yscale,
                    xscale="log",
                    is_only_mse=is_only_mse,
                )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
