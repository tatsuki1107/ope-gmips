from typing import Optional

import numpy as np
from sklearn.utils import check_random_state

from dataset import parse_behavior


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
