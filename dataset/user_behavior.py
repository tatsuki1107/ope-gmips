from typing import Optional
from typing import Union

import numpy as np
from obp.utils import softmax
from sklearn.utils import check_random_state

from utils import BEHAVIOR_PATTERNS


def parse_behavior(behavior: str) -> Union[tuple[str, int], None]:
    """parse string of a behavior name.

    Args:
        behavior: str
            string of a behavior name

    Returns:
        Union[tuple[str, int], None]: if the behavior name is successfully parsed,
        return a tuple of behavior name and an integer. Otherwise, return None.
    """

    for behavior_name, pattern in BEHAVIOR_PATTERNS.items():
        match = pattern.match(behavior)
        if match:
            k = int(match.group(1))
            return behavior_name, k
    return None


def create_decay_matrix(len_list: int) -> np.ndarray:
    """create a decay matrix of ranking list.

    Args:
        len_list: int
            length of a recommendation list

    Returns:
        np.ndarray: decay matrix of ranking list
    """

    position = np.arange(len_list, dtype=float)
    tiled_position = np.tile(position, reps=(len_list, 1))
    repeat_position = np.repeat(position[:, None], len_list, axis=1)
    decay_matrix = np.abs(tiled_position - repeat_position)
    np.fill_diagonal(decay_matrix, 1)

    return decay_matrix


def create_interaction_params(
    behavior_names: list[str],
    len_list: int,
    interaction_noise: float,
    random_state: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """create interaction parameters (user behavior matrix c) for each behavior.

    Args:
        behavior_names: list[str]
            list of behavior names

        len_list: int
            length of a recommendation list

        interaction_noise: float
            noise level of interaction parameters for each position.

        random_state: Optional[int] = None
            random seed.

    Returns:
        dict[str, np.ndarray]: dictionary of interaction parameters for each behavior.
    """

    random_ = check_random_state(random_state)
    interaction_params = random_.uniform(0, interaction_noise, size=(len_list, len_list))
    np.fill_diagonal(interaction_params, 1)

    decay_matrix = create_decay_matrix(len_list=len_list)
    interaction_params /= decay_matrix

    interaction_params_dict = dict()
    for behavior_name in behavior_names:
        if behavior_name == "standard":
            behavior_matrix = np.ones((len_list, len_list))

        elif behavior_name == "independent":
            behavior_matrix = np.eye(len_list)

        elif behavior_name == "cascade":
            behavior_matrix = np.tri(len_list)

        elif behavior_name == "inverse_cascade":
            behavior_matrix = np.tri(len_list).T

        else:
            parsed_behavior = parse_behavior(behavior_name)
            if parsed_behavior is None:
                raise NotImplementedError(f"You should implement the behavior: {behavior_name} to parse.")

            behavior_name_, int_ = parsed_behavior
            if behavior_name_ == "neighbor_k":
                neighbor_k = int_
                all_one_mat = np.ones((len_list, len_list), dtype=int)
                behavior_matrix = np.triu(all_one_mat, k=-neighbor_k) & np.tril(all_one_mat, k=neighbor_k)

            elif behavior_name_ == "top_k_cascade":
                top_k = int_
                behavior_matrix = np.ones((len_list, len_list), dtype=int)
                behavior_matrix[:, :top_k] = 1
                np.fill_diagonal(behavior_matrix, 1)
                behavior_matrix = behavior_matrix & np.tri(len_list, dtype=int)

            elif behavior_name_ == "random_c":
                random_state_ = int_
                behavior_matrix = check_random_state(random_state_).randint(2, size=(len_list, len_list))
                np.fill_diagonal(behavior_matrix, 1)

            else:
                raise NotImplementedError(
                    f"You should implement the behavior: {behavior_name} to create behavior matrix."
                )

        interaction_params_dict[behavior_name] = behavior_matrix * interaction_params

    return interaction_params_dict


def action_interaction_reward_function(
    base_expected_reward_factual: np.ndarray,
    user_behavior: np.ndarray,
    interaction_params: dict[str, np.ndarray],
) -> np.ndarray:
    """calculate the expected reward of each position in a recommendation list.

    Args:
        base_expected_reward_factual: np.ndarray
            base expected reward of each position in a recommendation list (i.e. \bar{q}).

        user_behavior: np.ndarray
            user behavior names of each round.

        interaction_params: dict[str, np.ndarray]
            user behavior matrix for each behavior.

    Returns:
        np.ndarray: expected reward of each position in a recommendation list.
    """

    len_list = base_expected_reward_factual.shape[1]

    expected_reward_factual_fixed = np.zeros_like(base_expected_reward_factual)
    for behavior_name in np.unique(user_behavior):
        behavior_mask = user_behavior == behavior_name

        for pos_ in range(len_list):
            q_x_a_k_fixed = (interaction_params[behavior_name][pos_] * base_expected_reward_factual[behavior_mask]).sum(
                axis=1
            )
            expected_reward_factual_fixed[behavior_mask, pos_] = q_x_a_k_fixed

    return expected_reward_factual_fixed


def linear_user_behavior_model(
    context: np.ndarray,
    gamma_z: Optional[np.ndarray],
    delta: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """linear model to create user behavior given context.

    Args:
        context: np.ndarray
            context of each round.

        gamma_z: Optional[np.ndarray]
            user behavior coefficients for each behavior. if None, only one behavior is observed.

        delta: float = 1.0
            parameter to control the strength of user behavior.

        random_state: Optional[int] = None
            random seed.

    Returns:
        np.ndarray: probability of each behavior given context.
    """

    if gamma_z is None:
        n_rounds = len(context)
        p_c_x = np.ones((n_rounds, 1))
    else:
        random_ = check_random_state(random_state)
        n_behavior_model, dim_context = len(gamma_z), context.shape[1]
        user_behavior_coef = random_.uniform(-1, 1, size=(dim_context, n_behavior_model))

        behavior_logits = np.abs(context @ user_behavior_coef)
        lambda_z = np.exp((2 * delta - 1) * gamma_z)
        p_c_x = softmax(lambda_z * behavior_logits)

    return p_c_x
