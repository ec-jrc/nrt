"""Model fitting

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return both coefficient (beta) and residuals matrices
These functions are meant to be called in ``nrt.BaseNrt._fit()``

Citations:

- Brooks, E.B., Wynne, R.H., Thomas, V.A., Blinn, C.E. and Coulston, J.W., 2013.
  On-the-fly massively multitemporal change detection using statistical quality
  control charts and Landsat data. IEEE Transactions on Geoscience and Remote Sensing,
  52(6), pp.3316-3332.
"""
import numpy as np
import numba
from nrt.stats import nanlstsq, mad, bisquare


def shewhart(X, y, L):
    """Fit an OLS model with outlier filtering

    As described in Brooks et al. 2014, following an initial OLS fit, outliers are
    identified using a shewhart control chart and removed. A second OLS fit is performed
    using the remaining partition of the data.

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ({(M,), (M, K)} np.ndarray): Matrix of dependant variables
        L (float): control limit used for outlier filtering. Must be a positive
            float. Lower values indicate stricter filtering

    Returns:
        beta (numpy.ndarray): The array of regression estimators
        residuals (numpy.ndarray): The array of residuals
    """
    beta_full, residuals_full = ols(X, y)
    # Shewhart chart to get rid of outliers (clouds etc)
    sigma = np.nanstd(residuals_full, axis=0)
    shewhart_mask = np.abs(residuals_full) > L * sigma
    y[shewhart_mask] = np.nan
    # fit again, but without outliers
    beta, residuals = ols(X, y)
    return beta, residuals


def ols(X, y):
    """Fit simple OLS model

    The array of dependant variables ``y`` may contain no data values denoted by
    ``np.nan``

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

    beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    resid = np.zeros(y.shape, dtype=np.float64)
    for idx in range(y.shape[1]):
        y_sub = y[:,idx]
        beta[:,idx], resid[:,idx] = _weight_fit(X, y_sub, np.ones_like(y_sub))
        scale = scale_est(resid[:,idx], c=scale_constant)

        EPS = np.finfo('float').eps
        if scale < EPS:
            continue

        iteration = 1
        converged = 0
        while not converged and iteration < maxiter:
            _beta = beta.copy()
            weights = M(resid[:,idx] / scale, c=tune)
            beta[:,idx], resid[:,idx] = _weight_fit(X, y_sub, weights)
            if update_scale:
                scale = max(EPS,
                                 scale_est(resid[:,idx], c=scale_constant))
            iteration += 1
            converged = not np.any(np.fabs(beta - _beta > tol))
    if is_1d:
        resid = resid.squeeze(axis=-1)
        beta = beta.squeeze(axis=-1)

    return beta, resid


# Broadcast on sw prevents nopython
# TODO: check implementation https://github.com/numba/numba/pull/1542
#@numba.jit()
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

    Xw = X * sw[:, None]
    yw = y * sw

    beta = nanlstsq(Xw, yw)

    resid = y - np.dot(X, beta)

    return beta, resid