from dataclasses import dataclass
from pathlib import Path
from pathlib import PosixPath
from typing import Optional

import numpy as np
from obp.dataset import BaseRealBanditDataset
from obp.utils import sample_action_fast
from obp.utils import softmax
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils import resample

from dataset.user_behavior import action_interaction_reward_function
from dataset.user_behavior import create_interaction_params
from dataset.user_behavior import linear_user_behavior_model
from utils import sample_slate_fast_with_replacement


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the probability given the logit.

    Args:
        x (np.ndarray): logit value

    Returns:
        np.ndarray: probability
    """
    return 1 / (1 + np.exp(-x))


@dataclass
class ExtremeBanditDatasetWithActionEmbed(BaseRealBanditDataset):
    """Extreme Dataset Class

    Args:
        n_components: int
            number of components for PCA.

        n_actions_at_k: int
            number of unique actions at psotion k.

        len_list: int
            length of a list of recommended items.

        reward_std: float
            standard deviation of the reward noise.

        behavior_params: dict[str, float]
            user behavior parameters. key is the name of the behavior and value is the temperature parameter.

        beta: float
            temperature parameter for the behavior policy function.

        interaction_noise: float = 10.0
            interaction noise for each position.

        dataset_name: str = "EUR-Lex4K"  # EUR-Lex4K or "RCV1-2K"
            name of the dataset. You can choose from "EUR-Lex4K" or "RCV1-2K".

        max_reward_noise: float = 0.2
            maximum reward noise for each action.

        random_state: int = 12345
            random seed.

        delta: float = 1.0
            parameter for the user behavior model.

    References:
        Saito, Yuta, Qingyang Ren, and Thorsten Joachims.
        "Off-policy evaluation for large action spaces via conjunct effect modeling.", 2023.

        Kiyohara, Haruka, Masahiro Nomura, and Yuta Saito.
        "Off-policy evaluation of slate bandit policies via optimizing abstraction.", 2024.
    """

    n_components: int
    n_actions_at_k: int
    len_list: int
    reward_std: float
    behavior_params: dict[str, float]
    beta: float
    interaction_noise: float = 10.0
    dataset_name: str = "EUR-Lex4K"  # EUR-Lex4K or "RCV1-2K"
    max_reward_noise: float = 0.2
    random_state: int = 12345
    delta: float = 1.0

    def __post_init__(self):
        self.data_path = Path().cwd() / "rawdata" / self.dataset_name
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        self.min_label_frequency = 1
        if self.dataset_name == "EUR-Lex4K":
            self.train_size, self.test_size = None, None
        elif self.dataset_name == "RCV1-2K":
            self.train_size, self.test_size = 16000, 4000
        self.random_ = check_random_state(self.random_state)

        self.load_raw_data()
        self.pre_process()

        # train a classifier to define a logging policy
        self._train_pi_b()

        # define the unknown distribution of the user behavior
        self.interaction_params = create_interaction_params(
            behavior_names=list(self.behavior_params.keys()),
            len_list=self.len_list,
            interaction_noise=self.interaction_noise,
            random_state=self.random_state,
        )

        self.gamma_z = np.array(list(self.behavior_params.values())) if len(self.behavior_params) > 1 else None
        # c ~ p(c|x)
        self.p_c_x = linear_user_behavior_model(
            context=self.train_contexts,
            gamma_z=self.gamma_z,
            delta=self.delta,
            random_state=self.random_state,
        )

        self.id2user_behavior = {i: c for i, (c, _) in enumerate(self.behavior_params.items())}

    def load_raw_data(self) -> None:
        """Load raw data from the dataset."""

        self.train_feature, self.train_label = self._load_raw_data(
            file_path=self.data_path / "train.txt", data_size=self.train_size
        )
        self.test_feature, self.test_label = self._load_raw_data(
            file_path=self.data_path / "test.txt", data_size=self.test_size
        )

    def _load_raw_data(self, file_path: PosixPath, data_size: Optional[int] = None) -> tuple[np.ndarray, ...]:
        """Load raw data from the train or test dataset.

        Args:
            file_path: PosixPath
                path to the dataset file.

            data_size: Optional[int] = None
                number of data to load. If None, load all data.

        Returns:
            tuple[np.ndarray, ...]: feature and label data.
        """

        with open(file_path) as file:
            num_data, num_feature, num_label = file.readline().split()
            num_data = int(num_data) if data_size is None else data_size
            num_feature, num_label = int(num_feature), int(num_label)

            feature, label = [], []
            for _ in range(num_data):
                data_ = file.readline().split(" ")
                label_ = [int(x) for x in data_[0].split(",") if x != ""]
                feature_index = [int(x.split(":")[0]) for x in data_[1:]]
                feature_ = [float(x.split(":")[1]) for x in data_[1:]]

                label.append(sp.csr_matrix(([1.0] * len(label_), label_, [0, len(label_)]), shape=(1, num_label)))
                feature.append(sp.csr_matrix((feature_, feature_index, [0, len(feature_)]), shape=(1, num_feature)))

        return sp.vstack(feature).toarray(), sp.vstack(label).toarray()

    def pre_process(self) -> None:
        """Preprocess the raw data to generate semi-synthetic data."""

        self.n_train, self.n_test = self.train_feature.shape[0], self.test_feature.shape[0]
        self.n_actions = self.n_actions_at_k * self.len_list

        # delete some rare actions
        all_label = sp.vstack([self.train_label, self.test_label]).astype(np.int8).toarray()
        idx = all_label.sum(axis=0) >= self.min_label_frequency
        all_label = all_label[:, idx]

        # extract candidate_actions
        self.n_labels = all_label.shape[1]
        random_ = check_random_state(self.random_state)
        self.candidate_labels = random_.choice(self.n_labels, size=self.n_actions, replace=False)
        self.candidate_action_set_at_k = np.arange(self.n_actions).reshape(self.len_list, self.n_actions_at_k)

        # generate reward_noise (depends on each action)
        random_ = check_random_state(self.random_state)
        self.eta = random_.uniform(self.max_reward_noise, size=self.n_actions)

        self.train_label_all = sp.csr_matrix(all_label[: self.n_train], dtype=np.float32).toarray()
        self.train_label = self.train_label_all[:, self.candidate_labels]
        self.base_train_expected_rewards = self.train_label * (1 - self.eta) + (1 - self.train_label) * (self.eta - 1)
        # self.base_train_expected_rewards = sigmoid(x=self.base_train_expected_rewards)
        self.base_train_expected_rewards = self.base_train_expected_rewards[:, self.candidate_action_set_at_k.T]

        self.test_label_all = sp.csr_matrix(all_label[self.n_train :], dtype=np.float32).toarray()
        self.test_label = self.test_label_all[:, self.candidate_labels]
        self.base_test_expected_rewards = self.test_label * (1 - self.eta) + (1 - self.test_label) * (self.eta - 1)
        # self.base_test_expected_rewards = sigmoid(x=self.base_test_expected_rewards)
        self.base_test_expected_rewards = self.base_test_expected_rewards[:, self.candidate_action_set_at_k.T]

        self.train_contexts = self.sc.fit_transform(self.pca.fit_transform(self.train_feature))
        self.test_contexts = self.sc.fit_transform(self.pca.fit_transform(self.test_feature))

        self.dim_context = self.train_contexts.shape[1]

    def _train_pi_b(self, max_iter: int = 500, batch_size: int = 2000) -> None:
        """Train a classifier to define a logging policy for each position.

        Args:
            max_iter: int = 500
                maximum number of iterations for the Ridge regression.

            batch_size: int = 2000
                batch size for the training.
        """

        idx = self.random_.choice(self.n_test, size=batch_size, replace=False)
        batch_context = self.test_contexts[idx]
        batch_expected_rewards = self.base_test_expected_rewards[idx]
        self.regressors = [
            MultiOutputRegressor(Ridge(max_iter=max_iter, random_state=self.random_state)) for _ in range(self.len_list)
        ]

        for pos_ in range(self.len_list):
            self.regressors[pos_].fit(batch_context, batch_expected_rewards[:, :, pos_])

        self.pi_b = np.zeros((self.n_train, self.n_actions_at_k, self.len_list))
        for pos_ in range(self.len_list):
            pi_b_k = self._compute_pi_b_k(regressor=self.regressors[pos_], contexts=self.train_contexts)
            self.pi_b[:, :, pos_] = pi_b_k

    def _compute_pi_b_k(self, regressor: MultiOutputRegressor, contexts: np.ndarray) -> np.ndarray:
        """Compute the action distribution of the logging policy for each position.

        Args:
            regressor: MultiOutputRegressor
                regressor to predict the expected reward.

            contexts: np.ndarray
                context data.

        Returns:
            np.ndarray: action distribution of the logging policy.
        """

        q_x_a_k_hat = regressor.predict(contexts)
        pi_b_k = softmax(self.beta * q_x_a_k_hat)
        return pi_b_k

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:
        """Obtain batch bandit feedback data.

        Args:
            n_rounds: int
                number of rounds to generate the data.

        Returns:
            dict: logged bandit feedback data.
        """

        contexts, p_c_x, pi_b, base_q_x_a_e = resample(
            self.train_contexts,
            self.p_c_x,
            self.pi_b,
            self.base_train_expected_rewards,
            replace=False,
            n_samples=n_rounds,
            random_state=self.random_,
        )

        user_behavior_id = sample_action_fast(p_c_x)
        user_behavior = np.array([self.id2user_behavior[i] for i in user_behavior_id])

        # a ~ \pi_b(\cdot|x)
        action_id_at_k, rankings = sample_slate_fast_with_replacement(
            pi_b, candidate_action_set_at_k=self.candidate_action_set_at_k
        )
        rounds, positions = np.arange(n_rounds)[:, None], np.arange(self.len_list)[None, :]
        pscores = pi_b[rounds, action_id_at_k, positions]

        base_q_x_a_e_factual = base_q_x_a_e[rounds, action_id_at_k, positions]

        q_x_a_e_factual = action_interaction_reward_function(
            base_expected_reward_factual=base_q_x_a_e_factual,
            user_behavior=user_behavior,
            interaction_params=self.interaction_params,
        )

        # r ~ p(r|x,a,e)
        rewards = self.random_.binomial(n=1, p=sigmoid(q_x_a_e_factual))

        return dict(
            n_rounds=n_rounds,
            n_actions_at_k=self.n_actions_at_k,
            len_list=self.len_list,
            context=contexts,
            action=rankings,
            reward=rewards,
            pscore=pscores,
            pi_b=pi_b,
            evaluation_policy_logit=base_q_x_a_e,
            action_id_at_k=action_id_at_k,
        )

    def calc_on_policy_policy_value(self, action_dist: np.ndarray) -> np.float64:
        """Calculate the on-policy policy value given the action distribution of the evaluation policy.

        Args:
            action_dist: np.ndarray
                action distribution of the evaluation policy.

        Returns:
            np.float64: approximate policy value.
        """

        user_behavior_id = sample_action_fast(self.p_c_x)
        user_behavior = np.array([self.id2user_behavior[i] for i in user_behavior_id])

        # a ~ \pi_e
        action_id_at_k = sample_slate_fast_with_replacement(action_dist)

        base_q_x_a_e_factual = self.base_train_expected_rewards[
            np.arange(self.n_train)[:, None], action_id_at_k, np.arange(self.len_list)[None, :]
        ]
        expected_reward_factual = action_interaction_reward_function(
            base_expected_reward_factual=base_q_x_a_e_factual,
            user_behavior=user_behavior,
            interaction_params=self.interaction_params,
        )
        expected_reward_factual = sigmoid(expected_reward_factual)

        return expected_reward_factual.sum(1).mean()
