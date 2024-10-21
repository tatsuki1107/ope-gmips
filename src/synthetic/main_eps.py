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
from ope import simulate_evaluation
from utils import TQDM_FORMAT
from utils import aggregate_simulation_results
from utils import visualize_mean_squared_error


cs = ConfigStore.instance()

logger = logging.getLogger(__name__)

ope_estimators = [
    SNIPS(estimator_name="snSIPS"),
    MIPS(estimator_name="MSIPS"),
    SNIPS(estimator_name="snIIPS"),
    MIPS(estimator_name="MIIPS"),
    SNIPS(estimator_name="snRIPS"),
    MIPS(estimator_name="MRIPS"),
]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start the experiment. the settings are as follows")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)

    for user_behavior in cfg.user_behavior:
        result_path = log_path / user_behavior
        result_path.mkdir(parents=True, exist_ok=True)

        behavior_params = {user_behavior: 1.0}
        result_df_list = []
        for eps in cfg.variation.eps_list:
            dataset = SyntheticRankingDatasetWithActionEmbeds(
                n_actions_at_k=cfg.n_unique_actions_at_k,
                dim_context=cfg.dim_context,
                n_cat_dim=cfg.n_cat_dim,
                n_cat_per_dim=cfg.n_cat_per_dim,
                n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
                n_deficient_actions_at_k=cfg.n_deficient_actions_at_k,
                len_list=cfg.len_list,
                behavior_params=behavior_params,
                random_state=cfg.random_state,
                reward_noise=cfg.reward_noise,
                interaction_noise=cfg.interaction_noise,
                beta=cfg.beta,
                eps=eps,
            )

            # calculate ground truth policy value (on policy)
            test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.test_size, is_online=True)
            policy_value = test_data["expected_reward_factual"].sum(1).mean()

            message = f"behavior={user_behavior}, eps={eps}"
            job_args = [(ope_estimators, dataset, cfg.val_size, eps) for _ in range(cfg.n_val_seeds)]
            with Pool(cpu_count() - 1) as pool:
                imap_iter = pool.imap(simulate_evaluation, job_args)
                tqdm_ = tqdm(imap_iter, total=cfg.n_val_seeds, desc=message, bar_format=TQDM_FORMAT)
                result_list = list(tqdm_)

            logger.info(tqdm_)
            # calculate MSE
            result_df = aggregate_simulation_results(
                simulation_result_list=result_list, policy_value=policy_value, x_value=eps
            )
            result_df_list.append(result_df)

        result_df = pd.concat(result_df_list).reset_index(level=0)
        result_df.to_csv(result_path / "result.csv")

        visualize_mean_squared_error(
            result_df=result_df,
            xlabel="epsilon of target policy",
            img_path=result_path,
            xscale="linear",
        )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
