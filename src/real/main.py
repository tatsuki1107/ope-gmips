import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import ExtremeBanditDatasetWithActionEmbed
from ope import InversePropensityScoreForRanking as IPS
from ope import MarginalizedIPSForRanking as MIPS
from ope import NNAbstractionLearnerWithSLOPE
from ope import RankingOffPolicyEvaluation
from ope import RankingOffPolicyEvaluationWithTune
from ope import UserBehaviorTree
from ope.importance_weight import NNAbstractionLearner
from ope.importance_weight import adaptive_weight
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
    MIPS(estimator_name="MIIPS"),
    MIPS(estimator_name="MRIPS"),
]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start the experiment. the settings are as follows")
    logger.info(cfg)

    log_path = Path(HydraConfig.get().run.dir)
    result_path = log_path / "mse_results"
    result_path.mkdir(parents=True, exist_ok=True)

    dataset = ExtremeBanditDatasetWithActionEmbed(
        dataset_name=cfg.dataset_name,
        n_components=cfg.dim_context,
        n_actions_at_k=cfg.n_unique_action_at_k,
        len_list=cfg.len_list,
        reward_std=cfg.reward_std,
        behavior_params=cfg.user_behaviors,
        beta=cfg.beta,
        random_state=cfg.random_state,
    )

    # calculate ground truth policy value (on policy)
    action_dist = gen_eps_greedy(dataset.base_train_expected_rewards, eps=cfg.eps)
    policy_value = dataset.calc_on_policy_policy_value(action_dist=action_dist)
    logger.info(f"policy value: {policy_value}")

    result_df_list = []
    for val_size in cfg.val_size_list:
        message = f"data={cfg.dataset_name}, val_size={val_size}"
        tqdm_ = tqdm(range(cfg.n_val_seeds), desc=message, bar_format=TQDM_FORMAT)
        result_list = []
        for seed in tqdm_:
            # generate synthetic data
            val_data = dataset.obtain_batch_bandit_feedback(n_rounds=val_size)

            # evaluation policy
            action_dist = gen_eps_greedy(expected_reward=val_data["evaluation_policy_logit"], eps=cfg.eps)

            if seed % 100 == 0:
                # learn and obtain the action embedding
                abstraction_learner = NNAbstractionLearner(
                    model_name="ActionEmbeddingModel",
                    dim_context=cfg.dim_context,
                    n_actions_at_k=cfg.n_unique_action_at_k,
                    len_list=cfg.len_list,
                    n_cat_dim=cfg.n_cat_dim,
                    n_cat_per_dim=cfg.n_cat_per_dim,
                    hidden_size=cfg.hidden_size,
                    learning_rate=cfg.learning_rate,
                    num_epochs=cfg.num_epochs,
                    batch_size=cfg.batch_size,
                    weight_decay=cfg.weight_decay,
                    loss_img_path=log_path / "abstraction_loss.png",
                    random_state=cfg.random_state,
                )
                unique_action_embeddings, _ = abstraction_learner.fit_predict(
                    context=val_data["context"],
                    action=val_data["action"],
                    pscore=val_data["pscore"],
                    action_id_at_k=val_data["action_id_at_k"],
                    is_discrete=True,
                )

                # define tuning estimators before ope
                ope_estimators_tune = [
                    NNAbstractionLearnerWithSLOPE(
                        model_name="ActionEmbeddingModel",
                        dim_context=cfg.dim_context,
                        n_actions_at_k=cfg.n_unique_action_at_k,
                        len_list=cfg.len_list,
                        n_cat_dim=cfg.n_cat_dim,
                        n_cat_per_dim=cfg.n_cat_per_dim,
                        hidden_size=cfg.hidden_size,
                        learning_rate=cfg.learning_rate,
                        num_epochs=cfg.num_epochs,
                        batch_size=cfg.batch_size,
                        weight_decay=cfg.weight_decay,
                        loss_img_path=log_path / "abstraction_loss_slope.png",
                        random_state=cfg.random_state,
                        estimator=MIPS(estimator_name="MRIPS (w/SLOPE)"),
                        param_name="estimated_action_embedding",
                        hyper_param=cfg.n_cat_dim_candidates,
                        lower_bound_func=estimate_student_t_lower_bound,
                        weight_func=marginalized_weight,
                    ),
                    UserBehaviorTree(
                        approximate_policy_value=policy_value,
                        estimator=IPS(estimator_name="AIPS (w/UBT)"),
                        param_name="estimated_user_behavior",
                        dataset=dataset,
                        weight_func=adaptive_weight,
                        candidate_weights=cfg.candidate_weights,
                        eps=cfg.eps,
                        len_list=cfg.len_list,
                        n_partition=cfg.n_partition,
                        min_samples_leaf=cfg.min_samples_leaf,
                        n_bootstrap=cfg.n_bootstrap,
                        max_depth=cfg.max_depth,
                        noise_level=cfg.noise_level,
                    ),
                ]
                ope_tune = RankingOffPolicyEvaluationWithTune(
                    ope_estimators_tune=[ope_estimators_tune[1]],
                )
                estimated_policy_values_with_tune = ope_tune.estimate_policy_values_with_tune(
                    bandit_feedback=val_data.copy(), action_dist=action_dist
                )
            else:
                estimated_policy_values_with_tune = ope_tune.estimate_policy_values_with_best_param(
                    bandit_feedback=val_data.copy(), action_dist=action_dist
                )

            val_data["unique_action_context"] = unique_action_embeddings
            val_data["action_context"] = unique_action_embeddings[
                np.arange(val_data["len_list"])[None, :], val_data["action_id_at_k"]
            ]
            val_data["observed_cat_dim"] = np.arange(cfg.n_cat_dim)

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
