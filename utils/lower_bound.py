import numpy as np
from scipy import stats


def estimate_student_t_lower_bound(x: np.ndarray, delta: float = 0.05, with_dev: bool = True) -> np.float64:
    """Estimate the lower bound of the mean of a random variable using the Student's t-distribution.

    Args:
        x: np.ndarray
            sample data to estimate the lower bound of the mean.

        delta: float = 0.05
            significance level.

        with_dev: bool = True
            flag to determine whether to return the confidence interval with the deviation.

    Returns:
        np.float64: estimated lower bound of the mean of a random variable.
    """

    n = x.shape[0]
    se = np.sqrt(np.var(x) / (n - 1))
    if with_dev:
        cnf = se * stats.t(n - 1).ppf(1.0 - (delta / 2))
        return cnf

    cnf = se * stats.t(n - 1).ppf(1.0 - delta)
    return np.mean(x) - cnf
