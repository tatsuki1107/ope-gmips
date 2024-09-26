from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state

from dataset import parse_behavior


def _compute_weight_by_specific_behavior(
    weight: np.ndarray, behavior_assumption: str, len_list: int, user_idx: np.ndarray
) -> np.ndarray:
    if behavior_assumption == "independent":
        return weight[user_idx]

    input_data = {"weight": weight, "len_list": len_list}
    if behavior_assumption in behavior_weight_dict:
        weight_func = behavior_weight_dict[behavior_assumption]

    elif behavior_assumption in complex_behavior_weight_dict:
        behavior_name_, int_ = parse_behavior(behavior_assumption)
        if behavior_name_ == "random_c":
            input_data["c"] = int_
        else:
            input_data["k"] = int_

        weight_func = complex_behavior_weight_dict[behavior_assumption]
    else:
        raise NotImplementedError

    weight = weight_func(**input_data)
    return weight[user_idx]


def _standard_weight(weight: np.ndarray, len_list: int) -> np.ndarray:
    stanadard_weight = weight.prod(1)
    stanadard_weight = np.tile(stanadard_weight[:, None], reps=len_list)

    return stanadard_weight


def _cascade_weight(weight: np.ndarray, len_list: int) -> np.ndarray:
    w_x_a_1_k = []
    for pos_ in range(len_list):
        w_x_a_1_k.append(weight[:, : pos_ + 1].prod(axis=1, keepdims=True))

    w_x_a_1_k = np.concatenate(w_x_a_1_k, axis=1)

    return w_x_a_1_k


def _inverse_cascade_weight(weight: np.ndarray, len_list: int) -> np.ndarray:
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
    top_k_cascade_iw = []
    for top_k_pos_ in range(k):
        top_k_cascade_iw.append(weight[:, : top_k_pos_ + 1].prod(axis=1, keepdims=True))

    for pos_ in range(k, len_list):
        top_k_cascade_iw.append(top_k_cascade_iw[-1] * weight[:, [pos_]])

    top_k_cascade_iw = np.concatenate(top_k_cascade_iw, axis=1)

    return top_k_cascade_iw


def _neighbor_k_weight(weight: np.ndarray, len_list: int, k: int) -> np.ndarray:
    w_x_a_neighbor_k = []
    for pos_ in range(len_list):
        top_k_neighbor, bottom_k_neighbor = max(pos_ - k, 0), min(pos_ + k, len_list) + 1
        w_x_a_neighbor_k.append(weight[:, top_k_neighbor:bottom_k_neighbor].prod(axis=1, keepdims=True))

    w_x_a_neighbor_k = np.concatenate(w_x_a_neighbor_k, axis=1)

    return w_x_a_neighbor_k


def _random_c_weight(weight: np.ndarray, len_list: int, c: int) -> np.ndarray:
    C_random = check_random_state(c).randint(2, size=(len_list, len_list))
    w_x_a_random_c = []
    for pos_ in range(len_list):
        w_x_a_random_c.append((weight * C_random[pos_]).prod(axis=1, keepdims=True))

    w_x_a_random_c = np.concatenate(w_x_a_random_c, axis=1)

    return w_x_a_random_c


complex_behavior_weight_dict = {
    "top_k_cascade": _top_k_cascade_weight,
    "neighbor_k": _neighbor_k_weight,
    "random_c": _random_c_weight,
}


def vanilla_weight(data: dict, action_dist: np.ndarray, behavior_assumption: str, **kwargs) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    position = np.arange(data["len_list"])[None, :]
    w_x_a_k = action_dist[rounds[:, None], data["action_id_at_k"], position] / data["pscore"]

    weight = _compute_weight_by_specific_behavior(
        weight=w_x_a_k, behavior_assumption=behavior_assumption, len_list=data["len_list"], user_idx=rounds
    )
    return weight


def adaptive_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
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
    action_embed_dim = kwargs["action_embed_dim"] if "action_embed_dim" in kwargs else data["observed_cat_dim"]
    rounds = np.arange(data["n_rounds"])

    if w_x_e_k is None:
        if "unique_action_context" in data:
            raise NotImplementedError
        else:
            reshaped_slate_embed = data["action_context"][:, :, None, None]
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


def marginalized_weight_hat(
    data: dict,
    action_dist: np.ndarray,
    behavior_assumption: str,
    pi_a_x_e_estimator: Optional[ClassifierMixin] = None,
    w_hat_x_e_k: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    if (w_hat_x_e_k is None) and (pi_a_x_e_estimator is None):
        raise ValueError

    action_embed_dim = kwargs["action_embed_dim"] if "action_embed_dim" in kwargs else data["observed_cat_dim"]
    rounds = np.arange(data["n_rounds"])

    if w_hat_x_e_k is None:
        w_x_a_k = action_dist / data["pi_b"]
        w_x_a_k = np.where(w_x_a_k < np.inf, w_x_a_k, 0.0)

        w_hat_x_e_k = []
        for pos_ in range(action_dist.shape[-1]):
            w_hat_x_e = _estimate_position_wise_w_x_e(
                w_x_a=w_x_a_k[:, :, pos_],
                context=data["context"],
                action=data["action_id_at_k"][:, pos_],
                action_embeds=data["action_context"][:, pos_, action_embed_dim],
                pi_a_x_e_estimator=pi_a_x_e_estimator,
            )
            w_hat_x_e_k.append(w_hat_x_e)

        w_hat_x_e_k = np.array(w_hat_x_e_k).T

    weight = _compute_weight_by_specific_behavior(
        weight=w_hat_x_e_k, behavior_assumption=behavior_assumption, len_list=data["len_list"], user_idx=rounds
    )
    return weight


def _estimate_position_wise_w_x_e(
    w_x_a: np.ndarray,
    context: np.ndarray,
    action: np.ndarray,
    action_embeds: np.ndarray,
    pi_a_x_e_estimator: ClassifierMixin,
) -> np.ndarray:
    n_rounds, n_actions_at_position = w_x_a.shape
    x_e = np.c_[context, action_embeds]
    pi_hat_a_x_e = np.zeros((n_rounds, n_actions_at_position))
    pi_a_x_e_estimator.fit(x_e, action)
    pi_hat_a_x_e[:, np.unique(action)] = pi_a_x_e_estimator.predict_proba(x_e)

    w_hat_x_e = (w_x_a * pi_hat_a_x_e).sum(1)
    return w_hat_x_e
