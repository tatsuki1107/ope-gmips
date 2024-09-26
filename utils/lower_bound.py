import numpy as np
from scipy import stats


def estimate_student_t_lower_bound(x: np.ndarray, delta: float = 0.05, with_dev: bool = True) -> np.float64:
    n = x.shape[0]
    se = np.sqrt(np.var(x) / (n - 1))
    if with_dev:
        cnf = se * stats.t(n - 1).ppf(1.0 - (delta / 2))
        return cnf

    cnf = se * stats.t(n - 1).ppf(1.0 - delta)
    return np.mean(x) - cnf
