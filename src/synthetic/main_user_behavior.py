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
from utils.aggregate import aggregate_simulation_results
from utils.dataset import SyntheticSlateDatasetWithActionEmbeds
from utils.estimator import IIPS
from utils.estimator import RIPS
from utils.estimator import SIPS
from utils.estimator import SlateOffPolicyEvaluation
from utils.plot import visualize_mean_squared_error
from utils.policy import gen_eps_greedy


TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


cs = ConfigStore.instance()
# cs.store(name="setting", node=Config)

logger = logging.getLogger(__name__)


ope_estimators = [
    IIPS(estimator_name="MIIPS", pscore_type="category"),
    RIPS(estimator_name="MRIPS", pscore_type="category"),
    SIPS(estimator_name="MSIPS", pscore_type="category"),
]


def calc_avg_reward(args: tuple[SyntheticSlateDatasetWithActionEmbeds, int]) -> np.float64:
    dataset, test_size = args

    test_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_size, is_online=True)
    avg_reward = test_data["expected_reward_factual"].sum(1).mean()
    return avg_reward


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start experiment...")

    log_path = Path(HydraConfig.get().run.dir)

    for user_behavior in cfg.user_behavior:
        if user_behavior == "all":
            continue

        result_path = log_path / user_behavior
        result_path.mkdir(parents=True, exist_ok=True)

        result_df_list = []
        for behavior_complexity in cfg.variation.behavior_complexity_list:
            behavior_complexity /= 2
            if user_behavior == "independent":
                behavior_ratio = {
                    "independent": 1 - behavior_complexity,
                    "cascade": behavior_complexity / 2,
                    "all": behavior_complexity / 2,
                }
            elif user_behavior == "cascade":
                behavior_ratio = {
                    "independent": behavior_complexity / 2,
                    "cascade": 1 - behavior_complexity,
                    "all": behavior_complexity / 2,
                }

            dataset = SyntheticSlateDatasetWithActionEmbeds(
                n_actions=cfg.n_unique_actions,
                dim_context=cfg.dim_context,
                n_cat_dim=cfg.n_cat_dim,
                n_cat_per_dim=cfg.n_cat_per_dim,
                n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
                len_list=cfg.len_list,
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

            message = f"behavior={user_behavior}, behavior ratio={behavior_ratio}"
            tqdm_ = tqdm(range(cfg.n_val_seeds), total=cfg.n_val_seeds, desc=message, bar_format=TQDM_FORMAT)
            result_list = []
            for _ in tqdm_:
                val_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.val_size)

                # evaluation policy
                evaluation_policy = gen_eps_greedy(val_data["evaluation_policy_logit"], eps=cfg.eps)
                evaluation_pscore_dict = dataset.aggregate_propensity_score(
                    pi=evaluation_policy,
                    slate_action=val_data["action"],
                    p_e_d_a=val_data["p_e_d_a"],
                    slate_action_context=val_data["action_context"],
                )

                # off policy evaluation
                ope = SlateOffPolicyEvaluation(
                    bandit_feedback=val_data,
                    ope_estimators=ope_estimators,
                )
                estimated_policy_values = ope.estimate_policy_values(
                    evaluation_policy_pscore_dict=evaluation_pscore_dict
                )

                result_list.append(estimated_policy_values)

            logger.info(tqdm_)
            # calculate MSE
            result_df = aggregate_simulation_results(
                simulation_result_list=result_list,
                policy_value=approximate_policy_value,
                x_value=behavior_complexity * 2,
            )
            result_df_list.append(result_df)

        result_df = pd.concat(result_df_list).reset_index(level=0)
        result_df.to_csv(result_path / "result.csv")

        visualize_mean_squared_error(
            result_df=result_df, xlabel="Complexity Of User Behavior", img_path=result_path / "mse.png", yscale="linear"
        )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
