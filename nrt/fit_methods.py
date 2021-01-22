"""Model fitting

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return both coefficient (beta) and residuals matrices
These functions are meant to be called in ``nrt.BaseNrt._fit()``

Citations:


"""
import numpy as np
import numba
from nrt.stats import nanlstsq, mad, bisquare, weighted_nanlstsq


def ols(X, y):
    """Fit simple OLS model

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ({(M,), (M, K)} np.ndarray): Matrix of dependant variables

    Returns:
        beta (numpy.ndarray): The array of regression estimators
        residuals (numpy.ndarray): The array of residuals
    """
    beta = nanlstsq(X, y)
    residuals = np.dot(X, beta) - y
    return beta, residuals


def rirls(X, y, M=bisquare, tune=4.685,
          scale_est=mad, scale_constant=0.6745,
          update_scale=True, maxiter=50, tol=1e-8, **kwargs):
    """ Robust Linear Model using Iterative Reweighted Least Squares (RIRLS)
    Perform robust fitting regression via iteratively reweighted least squares
    according to weight function and tuning parameter.
    Basically a clone from `statsmodels` that should be much faster and follows
    the scikit-learn __init__/fit/predict paradigm.
    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D independent variable
        scale_est (callable): function for scaling residuals
        tune (float): tuning constant for scale estimate
        maxiter (int, optional): maximum number of iterations (default: 50)
        tol (float, optional): convergence tolerance of estimate
            (default: 1e-8)
        scale_est (callable): estimate used to scale the weights
            (default: `mad` for median absolute deviation)
        scale_constant (float): normalization constant (default: 0.6745)
        update_scale (bool, optional): update scale estimate for weights
            across iterations (default: True)
        M (callable): function for scaling residuals
        tune (float): tuning constant for scale estimate
    Returns:
        tuple: beta-coefficients and residual vector
    """
    is_1d = y.ndim == 1
    if is_1d:
        y = y[:, np.newaxis]

    beta, resid = _weight_fit(X, y, np.ones_like(y))
    scale = scale_est(resid, c=scale_constant)

    EPS = np.finfo('float').eps

    converged = np.zeros_like(scale).astype('bool')
    converged[scale < EPS] = True

    iteration = 1
    while not all(converged) and iteration < maxiter:

        y_sub = y[:, ~converged]
        _beta = beta.copy()
        weights = M(resid[:, ~converged] / scale, c=tune)

        beta[:, ~converged], resid[:, ~converged] = _weight_fit(X, y_sub,
                                                                weights)
        iteration += 1
        is_converged = ~np.any(np.fabs(beta - _beta > tol), axis=0)
        converged[is_converged] = True
        if update_scale:
            est = scale_est(resid[:, ~converged], c=scale_constant)
            scale = np.where(EPS > est, EPS, est)

    if is_1d:
        resid = resid.squeeze(axis=-1)
        beta = beta.squeeze(axis=-1)

    return beta, resid


# np.tile prevents nopython could be changed so that weighting of X happens in
# weighted_nanlstsq()
# TODO: check implementation https://github.com/numba/numba/pull/1542
# @numba.jit()
def _weight_fit(X, y, w):
    """
    Apply a weighted OLS fit to data
    Args:
        X (ndarray): independent variables
        y (ndarray): dependent variable
        w (ndarray): observation weights
    Returns:
        tuple: coefficients and residual vector
    """
    sw = np.sqrt(w)
    X_big = np.tile(X, (y.shape[1], 1, 1))
    Xw = X_big * sw.T[:, :, None]
    yw = y * sw

    beta = weighted_nanlstsq(Xw, yw)

    resid = y - np.dot(X, beta)

    return beta, resid
