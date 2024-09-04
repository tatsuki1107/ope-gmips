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
from ope import SelfNormalizedAdaptiveIPS as SNAIPS
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
    SIPS(estimator_name="MSIPS (true)", pscore_type="category"),
    IIPS(estimator_name="MIIPS (true)", pscore_type="category"),
    RIPS(estimator_name="MRIPS (true)", pscore_type="category"),
    SIPS(estimator_name="MSIPS", pscore_type="category"),
    IIPS(estimator_name="MIIPS", pscore_type="category"),
    RIPS(estimator_name="MRIPS", pscore_type="category"),
    SNAIPS(estimator_name="snAIPS (true)", pscore_type="action"),
]


def compute_behavior_ratio(user_behavior: str, p: float) -> dict:
    behavior_set = {"independent", "cascade"}
    if user_behavior not in behavior_set:
        raise NotImplementedError

    behavior_ratio = dict()
    behavior_ratio["standard"] = p
    behavior_ratio[user_behavior] = 1 - p
    behavior_set.remove(user_behavior)
    behavior_ratio[behavior_set.pop()] = 0.0

    return behavior_ratio


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start experiment. the configure is as follow")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)

    for user_behavior in cfg.user_behavior:
        if user_behavior == "standard":
            continue

        result_path = log_path / user_behavior
        result_path.mkdir(parents=True, exist_ok=True)

        result_df_list = []
        for behavior_complexity in cfg.variation.behavior_complexity_list:
            behavior_ratio = compute_behavior_ratio(user_behavior, behavior_complexity)

            dataset = SyntheticSlateDatasetWithActionEmbeds(
                n_actions_at_k=cfg.variation.n_unique_actions_at_k,
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
            message = "calculate approximate policy value ..."
            num_workers = cpu_count() - 1
            with Pool(num_workers) as pool:
                imap_iter = pool.imap(calc_avg_reward, job_args)
                tqdm_ = tqdm(imap_iter, total=cfg.n_test_seeds, desc=message, bar_format=TQDM_FORMAT)
                approximate_policy_value = np.mean(list(tqdm_))

            logger.info(tqdm_)

            message = f"behavior={user_behavior}, behavior ratio={behavior_ratio}"
            job_args = [(ope_estimators, dataset, cfg.val_size, cfg.eps) for _ in range(cfg.n_val_seeds)]
            with Pool(num_workers) as pool:
                imap_iter = pool.imap(simulate_evaluation, job_args)
                tqdm_ = tqdm(imap_iter, total=cfg.n_val_seeds, desc=message, bar_format=TQDM_FORMAT)
                result_list = list(tqdm_)

            logger.info(tqdm_)
            # calculate MSE
            result_df = aggregate_simulation_results(
                simulation_result_list=result_list,
                policy_value=approximate_policy_value,
                x_value=behavior_complexity,
            )
            result_df_list.append(result_df)

        result_df = pd.concat(result_df_list).reset_index(level=0)
        result_df.to_csv(result_path / "result.csv")

        for yscale in ["linear", "log"]:
            for is_only_mse in [True, False]:
                img_path = result_path / f"{yscale}_mse_only={is_only_mse}_varying=behavior_{user_behavior}.png"
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
