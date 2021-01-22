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
from nrt.stats import nanlstsq, mad, bisquare, weighted_nanlstsq


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


# Broadcast on sw prevents nopython
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


def is_stable_ccdc(slope, residuals, threshold):
    """
    Check the stability of the fitted model using CCDC Method

    Stability is given if:
        1.             slope / RMSE < threshold
        2. first observation / RMSE < threshold
        3.  last observation / RMSE < threshold

    For multiple bands Zhu et al. 2014 proposed the mean of all bands to
    be > 1 to signal instability.

    Args:
        slope (ndarray): 1D slope/trend of coefficients
        residuals (ndarray): 2D corresponding residuals
        threshold (float): threshold value to signal change
    Returns:
        ndarray: 1D boolean array with True = stable
    """
    # TODO check if SWIR and Green are the same size
    # "flat" 2D implementation
    rmse = np.sqrt(np.nanmean(residuals ** 2, axis=0))
    slope_rmse = slope / rmse < threshold
    first = residuals[0, :] / rmse < threshold
    last = residuals[-1, :] / rmse < threshold
    print(first)

    # It's only stable if all conditions are met
    is_stable = slope_rmse & first & last

    return is_stable


def screen_outliers_rirls(X, green, swir, **kwargs):
    """
    Screen for missed clouds and other outliers using green and SWIR band

    Args:
        X (ndarray): Design Matrix
        green (ndarray): 2D array containing spectral values
        swir (float): 2D array containing spectral values (~1.55-1.75um)
        **kwargs: arguments to be passed to fit_methods.rirls()
    Returns:
        ndarray: 2D (flat) boolean array with True = clear
    """
    # green and swir probably need to be flattened

    is_clear = ~np.isnan(green)

    # 1. estimate time series model using rirls for green and swir
    # TODO could be sped up, since masking is the same for green and swir
    g_beta, g_residuals = rirls(X, green, **kwargs)
    s_beta, s_residuals = rirls(X, swir, **kwargs)

    # Update mask using thresholds
    is_clear[g_residuals > 0.04] = False
    is_clear[s_residuals < -0.04] = False

    return is_clear
