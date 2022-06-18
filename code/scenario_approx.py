import numpy as np
from scipy import stats


def get_scenario_approx_constraints(
    Gamma: np.ndarray,
    Beta: np.ndarray,
    A: np.ndarray,
    samples: np.ndarray,
    include_slack: bool = False,
    eta: float = 0.01,
):
    """Assembles scenario approximation of chance constraint of a form
        `Gamma x <= Beta - A \chi`, where `x` is a control variable, `\chi` is a random variable

    Args:
        Gamma (numpy.ndarray): 2d array of normals. See description
        Beta (numpy.ndarray): 1d array of constants. See description
        A (numpy.ndarray): 2d array. See description
        samples (numpy.ndarray: scenarios or realization of random variable `\chi` to be used in scenario approximation
        include_slack (bool): if to include \cO - slack constraints
        eta: (float): confidence level for original chance constraint
    Returns:
        tuple: matrix of normals and vector of constants for scenario approximation
    """

    out_Gamma = np.concatenate([Gamma for i in range(samples.shape[0])], axis=0)
    out_Beta = np.concatenate(
        [Beta - A.dot(samples[i]) for i in range(samples.shape[0])], axis=0
    )
    if include_slack:
        Phi_inv = stats.norm.ppf(eta)
        Beta_O = Beta + Phi_inv
        out_Gamma = np.concatenate([out_Gamma, Gamma], axis=0)
        out_Beta = np.concatenate([out_Beta, Beta_O], axis=0)
    return out_Gamma, out_Beta
