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
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataset.user_behavior import action_interaction_reward_function
from dataset.user_behavior import create_interaction_params
from dataset.user_behavior import linear_user_behavior_model
from utils import sample_slate_fast_with_replacement
from utils import visualize_train_curve_of_abstraction_model


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

        hidden_size: int = 5
            hidden size for the abstraction model.

        n_cat_dim: int = 5
            number of dimension for the action embedding.

        n_cat_per_dim: int = 5
            number of categories per dimension for the action embedding.

        learning_rate: float = 0.001
            learning rate for the abstraction model.

        num_epochs: int = 10
            number of epochs for the abstraction model.

        batch_size: int = 10
            batch size for the abstraction model.

        weight_decay: float = 0.01
            parameter of L2 norm for the abstraction model.

    References:
        Saito, Yuta, Qingyang Ren, and Thorsten Joachims.
        "Off-policy evaluation for large action spaces via conjunct effect modeling.", 2023.

        Kiyohara, Haruka, Masahiro Nomura, and Yuta Saito.
        "Off-policy evaluation of slate bandit policies via optimizing abstraction.", 2024.
    """

    n_components: int
    n_actions_at_k: int
    len_list: int
    behavior_params: dict[str, float]
    beta: float
    interaction_noise: float = 15.0
    dataset_name: str = "EUR-Lex4K"  # EUR-Lex4K or "RCV1-2K"
    max_reward_noise: float = 0.2
    random_state: int = 12345
    delta: float = 1.0
    # abstraction settings
    hidden_size: int = 5
    n_cat_dim: int = 5
    n_cat_per_dim: int = 5
    learning_rate: float = 0.001
    num_epochs: int = 10
    batch_size: int = 10
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
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

        # train a abstraction model to obtain action embeddings
        self._train_abstraction_model()

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
        self.base_train_expected_rewards = self.base_train_expected_rewards[:, self.candidate_action_set_at_k.T]

        self.test_label_all = sp.csr_matrix(all_label[self.n_train :], dtype=np.float32).toarray()
        self.test_label = self.test_label_all[:, self.candidate_labels]
        self.base_test_expected_rewards = self.test_label * (1 - self.eta) + (1 - self.test_label) * (self.eta - 1)
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

    def _train_abstraction_model(self) -> None:
        """Train a abstraction model to obtain deterministic action embeddings in advance."""

        self.abstraction_learner = NNAbstractionLearner(
            model_name="ActionEmbeddingModel",
            dim_context=self.dim_context,
            n_actions_at_k=self.n_actions_at_k,
            len_list=self.len_list,
            n_cat_dim=self.n_cat_dim,
            n_cat_per_dim=self.n_cat_per_dim,
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            loss_img_path=self.data_path / "abstraction_loss.png",
            random_state=self.random_state,
        )
        self.abstraction_learner.fit(context=self.test_contexts, action=self.test_label)

        self.unique_action_context = self.abstraction_learner.obtain_action_embedding()

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
        action_context = self.unique_action_context[positions, action_id_at_k]

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
            action_context=action_context,
            reward=rewards,
            pscore=pscores,
            pi_b=pi_b,
            evaluation_policy_logit=base_q_x_a_e,
            action_id_at_k=action_id_at_k,
            unique_action_context=self.unique_action_context,
            observed_cat_dim=np.arange(self.n_cat_dim),
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

        return expected_reward_factual.mean(0)


@dataclass
class ContextActionDataset(Dataset):
    """Dataset class for the context-action pair to train the abstraction model.

    Args:
        context: np.ndarray
            context data.

        action: np.ndarray
            action data.
    """

    context: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        self.context = torch.tensor(self.context, dtype=torch.float32)
        self.action = torch.tensor(self.action, dtype=torch.float32)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return self.context[idx], self.action[idx]


class ActionEmbeddingModel(nn.Module):
    """Action Embedding Model. This model is defined the three-layer neural network which the middle layer is
    for the action embedding."""

    def __init__(self, dim_context: int, n_actions: int, n_cat_dim: int, hidden_size: int, random_state: int) -> None:
        super().__init__()
        torch.manual_seed(random_state)
        self.dim_context = dim_context
        self.n_actions = n_actions
        self.n_cat_dim = n_cat_dim
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.dim_context, self.hidden_size)
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding(self.n_actions, self.n_cat_dim)
        self.fc2 = nn.Linear(self.hidden_size + self.n_cat_dim, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = context.size(0)
        x1 = self.fc1(context)
        x1 = self.relu(x1)

        actions = torch.arange(self.n_actions)
        action_embed = self.embedding(actions)

        x1 = x1.unsqueeze(1).repeat(1, self.n_actions, 1)
        action_embed = action_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        x_combined = torch.cat([x1, action_embed], dim=2)
        logits = self.fc2(x_combined).squeeze(2)
        return logits


@dataclass
class NNAbstractionLearner:
    """Abstraction Learner class to obtain action embeddings.

    Args:
        model_name: str
            Action Embedding Model which is used to obtain action embeddings.

        dim_context: int
            dimension of the context data.

        n_actions_at_k: int
            number of unique actions at position k.

        len_list: int
            length of rankings

        n_cat_dim: int
            number of embedding dimensions.

        n_cat_per_dim: int
            number of categories per dimension.

        hidden_size: int
            hidden size for the abstraction model.

        learning_rate: float
            learning rate for the abstraction model.

        num_epochs: int
            number of epochs for the abstraction model.

        batch_size: int
            batch size for the abstraction model.

        weight_decay: int
            parameter of L2 norm for the abstraction model.

        loss_img_path: PosixPath
            path to save the training curve of the abstraction model.

        random_state: int = 12345
            random seed.
    """

    model_name: str
    dim_context: int
    n_actions_at_k: int
    len_list: int
    n_cat_dim: int
    n_cat_per_dim: int
    hidden_size: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    weight_decay: int
    loss_img_path: PosixPath
    random_state: int

    def __post_init__(self) -> None:
        # init
        self._init_model(n_cat_dim=self.n_cat_dim)

    def _init_model(self, n_cat_dim: int) -> None:
        self.n_actions = self.n_actions_at_k * self.len_list
        if self.model_name == "ActionEmbeddingModel":
            self.model = ActionEmbeddingModel(
                dim_context=self.dim_context,
                n_actions=self.n_actions,
                n_cat_dim=n_cat_dim,
                hidden_size=self.hidden_size,
                random_state=self.random_state,
            )
        else:
            raise NotImplementedError(f"model_name: {self.model_name} is not implemented.")

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.train_loss, self.val_loss = [], []
        self.n_cat_dim = n_cat_dim

    def _create_input_data(self, context: np.ndarray, action: np.ndarray, val_size: int) -> tuple[DataLoader]:
        """Devide the input data into the training and validation set.

        Args:
            context: np.ndarray
                context data.

            action: np.ndarray
                action data.

            pscore: np.ndarray
                propensity score of the logging policy at position k.

            val_size: int
                size of the validation set.

        Returns:
            tuple[DataLoader]: training and validation dataloader.
        """

        context_train, context_val, action_train, action_val = train_test_split(
            context, action, test_size=val_size, random_state=self.random_state
        )

        train_dataset = ContextActionDataset(context=context_train, action=action_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = ContextActionDataset(context=context_val, action=action_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader

    def fit(self, context: np.ndarray, action: np.ndarray, val_size: int = 0.2) -> None:
        """Fit the abstraction model to obtain action embeddings.

        Args:
            context: np.ndarray
                context data.

            action: np.ndarray
                action data.

            pscore: np.ndarray
                propensity score of the logging policy at position k.

            val_size: int = 0.2
                size of the validation set.
        """

        train_dataloader, val_dataloader = self._create_input_data(context=context, action=action, val_size=val_size)

        for _ in range(self.num_epochs):
            self.model.train()
            train_loss_epoch_ = []
            for contexts, actions in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(contexts)

                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()
                train_loss_epoch_.append(loss.item())

            self.train_loss.append(np.mean(train_loss_epoch_))

            self.model.eval()
            val_loss_epoch_ = []
            with torch.no_grad():
                for contexts, actions in val_dataloader:
                    outputs = self.model(contexts)
                    loss = self.criterion(outputs, actions)
                    val_loss_epoch_.append(loss.item())

            self.val_loss.append(np.mean(val_loss_epoch_))

        # health check
        visualize_train_curve_of_abstraction_model(
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            img_path=self.loss_img_path,
        )

    def obtain_action_embedding(self, is_discrete: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Obtain action embeddings from the abstraction model.

        Args:
            action_id_at_k: np.ndarray
                action id at position k.

            is_discrete: bool = True
                whether to discretize the action embeddings.

        Returns:
            np.ndarray: unique action embeddings and action embeddings.
        """

        unique_action_embeddings = self.model.embedding.weight.data.numpy()

        if is_discrete:
            bins = [self.n_cat_per_dim] * self.n_cat_dim
            discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="uniform")
            unique_action_embeddings = discretizer.fit_transform(unique_action_embeddings)

        unique_action_embeddings = unique_action_embeddings.reshape(self.len_list, self.n_actions_at_k, self.n_cat_dim)

        return unique_action_embeddings
