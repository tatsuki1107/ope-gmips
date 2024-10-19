import numpy as np


def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.3,
) -> np.ndarray:
    """Generate an evaluation policy via the epsilon-greedy rule.

    Args:
        expected_reward: np.ndarray
            expected reward of each action.

        is_optimal: bool, default=True
            whether the optimal action is selected or not.

        eps: float, default=0.3
            probability of selecting a random action. the smaller this value, the more deterministic the policy.

    """
    base_pol = np.zeros_like(expected_reward)
    a = np.argmax(expected_reward, axis=1) if is_optimal else np.argmin(expected_reward, axis=1)
    base_pol[
        np.arange(expected_reward.shape[0])[:, None],
        a,
        np.arange(expected_reward.shape[2])[None, :],
    ] = 1
    pol = (1.0 - eps) * base_pol
    pol += eps / expected_reward.shape[1]

    return pol
