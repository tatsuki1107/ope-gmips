from dataclasses import dataclass
from pathlib import PosixPath
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataset import parse_behavior
from utils import visualize_train_curve_of_abstraction_model


def _compute_weight_by_specific_behavior(
    weight: np.ndarray, behavior_assumption: str, len_list: int, user_idx: np.ndarray
) -> np.ndarray:
    """compute importance weight based on the behavior assumption.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        behavior_assumption: str
            behavior assumption to compute importance weight.

        len_list: int
            length of a recommendation list.

        user_idx: np.ndarray
            user index to compute importance weight.

    Returns:
        np.ndarray: importance weight based on the behavior assumption.
    """

    if behavior_assumption == "independent":
        return weight[user_idx]

    input_data = {"weight": weight, "len_list": len_list}
    if behavior_assumption in behavior_weight_dict:
        weight_func = behavior_weight_dict[behavior_assumption]

    else:
        behavior_assumption_, int_ = parse_behavior(behavior_assumption)
        if behavior_assumption_ == "random_c":
            input_data["c"] = int_

        elif behavior_assumption_ in {"top_k_cascade", "neighbor_k"}:
            input_data["k"] = int_

        else:
            raise NotImplementedError(f"behavior assumption {behavior_assumption} is not implemented.")

        weight_func = complex_behavior_weight_dict[behavior_assumption_]

    weight = weight_func(**input_data)
    return weight[user_idx]


def _standard_weight(weight: np.ndarray, len_list: int) -> np.ndarray:
    """compute (maginalized) standard importance weight.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        len_list: int
            length of a recommendation list.

    Returns:
        np.ndarray: (marginalized) standard importance weight.
    """

    stanadard_weight = weight.prod(1)
    stanadard_weight = np.tile(stanadard_weight[:, None], reps=len_list)

    return stanadard_weight


def _cascade_weight(weight: np.ndarray, len_list: int) -> np.ndarray:
    """compute (marginalized) cascade importance weight.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        len_list: int
            length of a recommendation list.

    Returns:
        np.ndarray: (marginalized) cascade importance weight.

    References:
        McInerney, James, et al.
        "Counterfactual evaluation of slate recommendations with sequential reward interactions.", 2020.

    """

    w_x_a_1_k = []
    for pos_ in range(len_list):
        w_x_a_1_k.append(weight[:, : pos_ + 1].prod(axis=1, keepdims=True))

    w_x_a_1_k = np.concatenate(w_x_a_1_k, axis=1)

    return w_x_a_1_k


def _inverse_cascade_weight(weight: np.ndarray, len_list: int) -> np.ndarray:
    """compute (marginalized) inverse cascade importance weight.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        len_list: int
            length of a recommendation list.

    Returns:
        np.ndarray: (marginalized) inverse cascade importance weight.

    References:
        McInerney, James, et al.
        "Counterfactual evaluation of slate recommendations with sequential reward interactions.", 2020.
    """

    w_x_a_k_K = []
    for pos_ in range(len_list):
        w_x_a_k_K.append(weight[:, pos_:].prod(axis=1, keepdims=True))

    w_x_a_k_K = np.concatenate(w_x_a_k_K, axis=1)

    return w_x_a_k_K


behavior_weight_dict = {
    "standard": _standard_weight,
    "cascade": _cascade_weight,
    "inverse_cascade": _inverse_cascade_weight,
}


def _top_k_cascade_weight(weight: np.ndarray, len_list: int, k: int) -> np.ndarray:
    """compute (marginalized) top-k cascade importance weight.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        len_list: int
            length of a recommendation list.

        k: int
            top-k integer value.

    Returns:
        np.ndarray: (marginalized) top-k cascade importance weight.

    References:
        Kiyohara, Haruka, et al.
        "Off-policy evaluation of ranking policies under diverse user behavior.", 2023.
    """

    top_k_cascade_iw = []
    for top_k_pos_ in range(k):
        top_k_cascade_iw.append(weight[:, : top_k_pos_ + 1].prod(axis=1, keepdims=True))

    for pos_ in range(k, len_list):
        top_k_cascade_iw.append(top_k_cascade_iw[-1] * weight[:, [pos_]])

    top_k_cascade_iw = np.concatenate(top_k_cascade_iw, axis=1)

    return top_k_cascade_iw


def _neighbor_k_weight(weight: np.ndarray, len_list: int, k: int) -> np.ndarray:
    """compute (marginalized) neighbor-k importance weight.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        len_list: int
            length of a recommendation list.

        k: int
            neighbor-k integer value.

    Returns:
        np.ndarray: (marginalized) neighbor-k importance weight.

    References:
        Kiyohara, Haruka, et al.
        "Off-policy evaluation of ranking policies under diverse user behavior.", 2023.
    """

    w_x_a_neighbor_k = []
    for pos_ in range(len_list):
        top_k_neighbor, bottom_k_neighbor = max(pos_ - k, 0), min(pos_ + k, len_list) + 1
        w_x_a_neighbor_k.append(weight[:, top_k_neighbor:bottom_k_neighbor].prod(axis=1, keepdims=True))

    w_x_a_neighbor_k = np.concatenate(w_x_a_neighbor_k, axis=1)

    return w_x_a_neighbor_k


def _random_c_weight(weight: np.ndarray, len_list: int, c: int) -> np.ndarray:
    """compute (marginalized) random behavior matrix importance weight given seed, c.

    Args:
        weight: np.ndarray
            vanilla importance weight or marginalized importance weight by action embeddings.

        len_list: int
            length of a recommendation list.

        c: int
            random seed.

    Returns:
        np.ndarray: (marginalized) random behavior matrix importance weight.
    """

    C_random = check_random_state(c).randint(2, size=(len_list, len_list))
    np.fill_diagonal(C_random, 1)
    C_random = C_random.astype(bool)

    w_x_a_random_c = []
    for pos_ in range(len_list):
        w_x_a_random_c.append((weight[:, C_random[pos_]]).prod(axis=1, keepdims=True))

    w_x_a_random_c = np.concatenate(w_x_a_random_c, axis=1)

    return w_x_a_random_c


complex_behavior_weight_dict = {
    "top_k_cascade": _top_k_cascade_weight,
    "neighbor_k": _neighbor_k_weight,
    "random_c": _random_c_weight,
}


def vanilla_weight(data: dict, action_dist: np.ndarray, behavior_assumption: str, **kwargs) -> np.ndarray:
    """compute vanilla importance weight.

    Args:
        data: dict
            bandit feedback data.

        action_dist: np.ndarray
            action distribution matrix of shape (n_rounds, n_actions_at_k, len_list).

        behavior_assumption: str
            behavior assumption to compute importance weight.

    Returns:
        np.ndarray: vanilla importance weight.
    """

    rounds = np.arange(data["n_rounds"])
    position = np.arange(data["len_list"])[None, :]
    w_x_a_k = action_dist[rounds[:, None], data["action_id_at_k"], position] / data["pscore"]

    weight = _compute_weight_by_specific_behavior(
        weight=w_x_a_k, behavior_assumption=behavior_assumption, len_list=data["len_list"], user_idx=rounds
    )
    return weight


def adaptive_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
    """compute adaptive importance weight.

    Args:
        data: dict
            bandit feedback data.

        action_dist: np.ndarray
            action distribution matrix of shape (n_rounds, n_actions_at_k, len_list).

    Returns:
        np.ndarray: adaptive importance weight.

    References:
        Kiyohara, Haruka, et al.
        "Off-policy evaluation of ranking policies under diverse user behavior.", 2023.
    """

    user_behavior = kwargs["estimated_user_behavior"] if "estimated_user_behavior" in kwargs else data["user_behavior"]

    rounds = np.arange(data["n_rounds"])
    position = np.arange(data["len_list"])[None, :]
    w_x_a_k = action_dist[rounds[:, None], data["action_id_at_k"], position] / data["pscore"]

    w_x_Phi_k_a_c = np.zeros_like(data["action"], dtype=float)
    for behavior_name in np.unique(user_behavior):
        behavior_mask = user_behavior == behavior_name
        weight_ = _compute_weight_by_specific_behavior(
            weight=w_x_a_k.copy(), behavior_assumption=behavior_name, len_list=data["len_list"], user_idx=behavior_mask
        )
        w_x_Phi_k_a_c[behavior_mask] = weight_

    return w_x_Phi_k_a_c


def marginalized_weight(
    data: dict, action_dist: np.ndarray, behavior_assumption: str, w_x_e_k: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    """compute marginalized importance weight.

    Args:
        data: dict
            bandit feedback data.

        action_dist: np.ndarray
            action distribution matrix of shape (n_rounds, n_actions_at_k, len_list).

        behavior_assumption: str
            behavior assumption to compute importance weight.

        w_x_e_k: Optional[np.ndarray] = None
            marginalized importance weight by action embeddings.
            if `w_x_e_k` is not None, marginalize based on `behavior_assumption`.

    Returns:
        np.ndarray: (doubly) marginalized importance weight.

    References:
        Saito, Yuta, and Thorsten Joachims.
        "Off-policy evaluation for large action spaces via embeddings.", 2022.

        Saito, Yuta, Qingyang Ren, and Thorsten Joachims.
        "Off-policy evaluation for large action spaces via conjunct effect modeling.", 2023.
    """

    action_embed_dim = kwargs["action_embed_dim"] if "action_embed_dim" in kwargs else data["observed_cat_dim"]
    rounds = np.arange(data["n_rounds"])

    if w_x_e_k is None:
        if "unique_action_context" in data:
            # `action_context` is a deterministic case.

            unique_action_context = data["unique_action_context"].transpose(1, 0, 2)
            # shape: (n_actions_at_k, len_list, n_cat_dim)

            action_context = data["action_context"][:, :, action_embed_dim]
            mask_e_a_e = []
            for e_a in unique_action_context[:, :, action_embed_dim]:
                mask_e_a_e.append(np.all(e_a == action_context, axis=2))

            mask_e_a_e = np.array(mask_e_a_e).transpose(1, 0, 2)
            p_e_k_x_pi_e = (action_dist * mask_e_a_e).sum(1)
            p_e_k_x_pi_b = (data["pi_b"] * mask_e_a_e).sum(1)

        else:
            # `action_context` is a stochastic case.

            reshaped_slate_embed = data["action_context"][:, :, None, None, action_embed_dim]
            position = np.arange(data["len_list"])
            action_at_k_idx = np.arange(data["n_actions_at_k"])[:, None, None]

            p_e_k_x_pi_e = np.ones_like(data["reward"], dtype=float)
            p_e_k_x_pi_b = np.ones_like(data["reward"], dtype=float)
            for pos_ in position:
                p_e_d_a_k = data["p_e_d_a_k"][pos_, action_at_k_idx, reshaped_slate_embed[:, pos_], action_embed_dim]
                p_e_d_a_k = p_e_d_a_k.squeeze(axis=2)

                # p(e|a) = \prod_{d=1}^{D} p(e(d)|a)
                p_e_a_k = p_e_d_a_k.prod(axis=2)

                p_e_k_x_pi_e[:, pos_] = (action_dist[:, :, pos_] * p_e_a_k).sum(1)
                p_e_k_x_pi_b[:, pos_] = (data["pi_b"][:, :, pos_] * p_e_a_k).sum(1)

        w_x_e_k = p_e_k_x_pi_e / p_e_k_x_pi_b

    weight = _compute_weight_by_specific_behavior(
        weight=w_x_e_k, behavior_assumption=behavior_assumption, len_list=data["len_list"], user_idx=rounds
    )
    return weight


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
    pscore: np.ndarray

    def __post_init__(self) -> None:
        self.context = torch.tensor(self.context, dtype=torch.float32)
        self.action = torch.tensor(self.action, dtype=torch.float32)
        self.pscore = torch.tensor(self.pscore, dtype=torch.float32)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return self.context[idx], self.action[idx], self.pscore[idx]


class ActionEmbeddingModel(nn.Module):
    """Action Embedding Model. This model is defined the three-layer neural network which the middle layer is
    for the action embedding."""

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
class NNAbstractionLearner:
    """Abstraction Learner class to obtain action embeddings.

    Args:
        model: ActionEmbeddingModel
            Action Embedding Model which is Three-layer neural network.

        n_cat_dim: int
            number of embedding dimensions.

        n_cat_per_dim: int
            number of categories per dimension.

        learning_rate: float
            learning rate for the abstraction model.

        num_epochs: int
            number of epochs for the abstraction model.

        batch_size: int
            batch size for the abstraction model.

        weight_decay: int
            parameter of L2 norm for the abstraction model.

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
            )
        else:
            raise NotImplementedError(f"model_name: {self.model_name} is not implemented.")

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.train_loss, self.val_loss = [], []
        self.n_cat_dim = n_cat_dim

    def _create_input_data(
        self, context: np.ndarray, action: np.ndarray, pscore: np.ndarray, val_size: int
    ) -> tuple[DataLoader]:
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

        context_ = np.repeat(context, repeats=self.len_list, axis=0)
        action_ = action.reshape(-1)[:, None]
        action_ = OneHotEncoder(sparse=False).fit_transform(action_)
        pscore_ = pscore.reshape(-1)[:, None]

        context_train, context_val, action_train, action_val, pscore_train, pscore_val = train_test_split(
            context_, action_, pscore_, test_size=val_size, random_state=self.random_state
        )

        train_dataset = ContextActionDataset(context=context_train, action=action_train, pscore=pscore_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = ContextActionDataset(context=context_val, action=action_val, pscore=pscore_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader

    def fit(self, context: np.ndarray, action: np.ndarray, pscore: np.ndarray, val_size: int = 0.2) -> None:
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

        train_dataloader, val_dataloader = self._create_input_data(
            context=context, action=action, pscore=pscore, val_size=val_size
        )

        for _ in range(self.num_epochs):
            self.model.train()
            train_loss_epoch_ = []
            for contexts, actions, pscores in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(contexts)

                weighted_actions = actions / pscores
                loss = self.criterion(outputs, weighted_actions)
                loss.backward()
                self.optimizer.step()
                train_loss_epoch_.append(loss.item())

            self.train_loss.append(np.mean(train_loss_epoch_))

            self.model.eval()
            val_loss_epoch_ = []
            with torch.no_grad():
                for contexts, actions, pscores in val_dataloader:
                    outputs = self.model(contexts)
                    weighted_actions = actions / pscores
                    loss = self.criterion(outputs, weighted_actions)
                    val_loss_epoch_.append(loss.item())

            self.val_loss.append(np.mean(val_loss_epoch_))

        # health check
        visualize_train_curve_of_abstraction_model(
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            img_path=self.loss_img_path,
        )

    def obtain_action_embedding(
        self, action_id_at_k: np.ndarray, is_discrete: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
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

        action_embeddings = unique_action_embeddings[np.arange(self.len_list)[None, :], action_id_at_k].copy()

        return unique_action_embeddings, action_embeddings

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_id_at_k: np.ndarray,
        val_size: int = 0.2,
        is_discrete: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the abstraction model and obtain action embeddings.

        Args:
            context: np.ndarray
                context data.

            action: np.ndarray
                action data.

            pscore: np.ndarray
                propensity score of the logging policy at position k.

            action_id_at_k: np.ndarray
                action id at position k.

            val_size: int = 0.2
                size of the validation set.

            is_discrete: bool = True
                whether to discretize the action embeddings.

        Returns:
            tuple[np.ndarray, np.ndarray]: unique action embeddings and action embeddings.
        """

        self.fit(context, action, pscore, val_size)
        return self.obtain_action_embedding(action_id_at_k, is_discrete)
