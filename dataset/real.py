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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import visualize_train_curve_of_abstraction_model


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@dataclass
class ExtremeBanditDatasetWithActionEmbed(BaseRealBanditDataset):
    n_components: int = 10
    reward_std: float = 3.0
    max_reward_noise: float = 0.2
    dataset_name: str = "EUR-Lex4K"  # EUR-Lex4K or "RCV1-2K"
    random_state: int = 12345
    # abstraction settings
    hidden_size: int = 5
    n_cat_dim: int = 5
    n_cat_per_dim: int = 5
    learning_rate: float = 0.001
    num_epochs: int = 10
    batch_size: int = 10
    weight_decay: float = 0.01

    def __post_init__(self):
        self.data_path = Path().cwd() / "rawdata" / self.dataset_name
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        self.min_label_frequency = 1
        if self.dataset_name == "EUR-Lex4K":
            self.train_size, self.test_size = None, None
        elif self.dataset_name == "RCV1-2K":
            self.train_size, self.test_size = 15000, 4000
        self.random_ = check_random_state(self.random_state)

        self.load_raw_data()
        self.pre_process()

        # train a classifier to define a logging policy
        self._train_pi_b()

        # train a abstraction model to obtain action embeddings
        self._train_abstraction_model()

    @property
    def n_actions(self) -> int:
        return int(self.train_label.shape[1])

    def load_raw_data(self) -> None:
        self.train_feature, self.train_label = self._load_raw_data(
            file_path=self.data_path / "train.txt", data_size=self.train_size
        )
        self.test_feature, self.test_label = self._load_raw_data(
            file_path=self.data_path / "test.txt", data_size=self.test_size
        )

    def _load_raw_data(self, file_path: PosixPath, data_size: Optional[int] = None) -> tuple[np.ndarray, ...]:
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
        self.n_train, self.n_test = self.train_feature.shape[0], self.test_feature.shape[0]

        # delete some rare actions
        all_label = sp.vstack([self.train_label, self.test_label]).astype(np.int8).toarray()
        idx = all_label.sum(axis=0) >= self.min_label_frequency
        all_label = all_label[:, idx]

        # generate reward_noise (depends on each action)
        self.eta = self.random_.uniform(self.max_reward_noise, size=all_label.shape[1])

        self.train_label = sp.csr_matrix(all_label[: self.n_train], dtype=np.float32).toarray()
        logits = self.train_label * (1 - self.eta) + (1 - self.train_label) * (self.eta - 1)
        self.train_expected_rewards = sigmoid(x=logits)

        self.test_label = sp.csr_matrix(all_label[self.n_train :], dtype=np.float32).toarray()
        logits = self.test_label * (1 - self.eta) + (1 - self.test_label) * (self.eta - 1)
        self.test_expected_rewards = sigmoid(x=logits)

        self.train_contexts = self.sc.fit_transform(self.pca.fit_transform(self.train_feature))
        self.test_contexts = self.sc.fit_transform(self.pca.fit_transform(self.test_feature))

        self.dim_context = self.train_contexts.shape[1]
        self.n_actions = self.train_label.shape[1]

    def _train_pi_b(self, max_iter: int = 500, batch_size: int = 2000) -> None:
        idx = self.random_.choice(self.n_test, size=batch_size, replace=False)
        self.regressor = MultiOutputRegressor(Ridge(max_iter=max_iter, random_state=self.random_state))
        self.regressor.fit(self.test_contexts[idx], self.test_expected_rewards[idx])

    def _train_abstraction_model(self) -> None:
        self.model = ActionEmbeddingModel(
            dim_context=self.dim_context,
            n_actions=self.n_actions,
            hidden_size=self.hidden_size,
            n_cat_dim=self.n_cat_dim,
        )

        self.abstraction_learner = AbstractionLearner(
            model=self.model,
            n_cat_dim=self.n_cat_dim,
            n_cat_per_dim=self.n_cat_per_dim,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
        )

        self.abstraction_learner.fit(context=self.test_contexts, action=self.test_label)

        # health check
        visualize_train_curve_of_abstraction_model(
            train_loss=self.abstraction_learner.train_loss,
            val_loss=self.abstraction_learner.val_loss,
            img_path=self.data_path / "img/abstraction_loss.png",
        )
        self.unique_action_context = self.abstraction_learner.obtain_action_embedding()

    def compute_pi_b(self, contexts: np.ndarray, beta: float = 1.0) -> np.ndarray:
        q_x_a_hat = self.regressor.predict(contexts)
        pi_b = softmax(beta * q_x_a_hat)
        return pi_b

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:
        idx = self.random_.choice(self.n_train, size=n_rounds, replace=False)
        contexts = self.train_contexts[idx]

        pi_b = self.compute_pi_b(contexts)
        # a ~ \pi_b(\cdot|x)
        actions = sample_action_fast(pi_b)

        # e_a
        action_context = self.unique_action_context[actions]

        q_x_a_e_factual = self.train_expected_rewards[idx, actions]
        # r ~ p(r|x,a,e)
        rewards = self.random_.normal(loc=q_x_a_e_factual, scale=self.reward_std)

        return dict(
            context=contexts,
            action=actions,
            action_context=action_context,
            reward=rewards,
            pscore=pi_b[:, actions],
            expected_reward=self.train_expected_rewards[idx],
            pi_b=pi_b,
            unique_action_context=self.unique_action_context,
        )

    @staticmethod
    def calc_ground_truth_policy_value(expected_reward: np.ndarray, action_dist: np.ndarray) -> np.float64:
        return np.average(expected_reward, weights=action_dist, axis=1).mean()


@dataclass
class ContextActionDataset(Dataset):
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
    def __init__(self, dim_context: int, n_actions: int, n_cat_dim: int, hidden_size: int) -> None:
        super().__init__()
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
class AbstractionLearner:
    model: ActionEmbeddingModel
    n_cat_dim: int
    n_cat_per_dim: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    weight_decay: int
    random_state: int = 12345

    def __post_init__(self) -> None:
        # init
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.train_loss, self.val_loss = [], []

    def _create_input_data(self, context: np.ndarray, action: np.ndarray, val_size: int) -> tuple[DataLoader]:
        context_train, context_val, action_train, action_val = train_test_split(
            context, action, test_size=val_size, random_state=self.random_state
        )

        train_dataset = ContextActionDataset(context=context_train, action=action_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = ContextActionDataset(context=context_val, action=action_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader

    def fit(self, context: np.ndarray, action: np.ndarray, val_size: int = 0.2) -> None:
        train_dataloader, val_dataloader = self._create_input_data(context=context, action=action, val_size=val_size)

        for _ in tqdm(range(self.num_epochs), desc="Training Abstraction Model"):
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

    def obtain_action_embedding(self, is_discrete: bool = True) -> np.ndarray:
        action_embeddings = self.model.embedding.weight.data.numpy()

        if is_discrete:
            discretizer = KBinsDiscretizer(n_bins=self.n_cat_per_dim, encode="ordinal", strategy="uniform")
            categorical_action_embedding = []
            for embedding_per_dim in action_embeddings.T:
                categorical_embedding_ = discretizer.fit_transform(embedding_per_dim.reshape(-1, 1)).flatten()
                categorical_action_embedding.append(categorical_embedding_)

            return np.array(categorical_action_embedding).T

        return action_embeddings
