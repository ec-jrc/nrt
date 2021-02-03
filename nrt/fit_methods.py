"""Model fitting

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return both coefficient (beta) and residuals matrices
These functions are meant to be called in ``nrt.BaseNrt._fit()``
"""
import numpy as np
import numba

from nrt.log import logger
from nrt.cusum import history_roc
from nrt.stats import nanlstsq, mad, bisquare, weighted_nanlstsq, is_stable_ccdc


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
    """Robust Linear Model using Iterative Reweighted Least Squares (RIRLS)

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

    beta, resid = weighted_ols(X, y, np.ones_like(y))
    scale = scale_est(resid, c=scale_constant)

    EPS = np.finfo('float').eps

    # Initializing array signaling converged models and setting everything
    # True where scale is smaller epsilon, to avoid singular matrices during
    # weighted fit
    converged = np.zeros_like(scale).astype('bool')
    converged[scale < EPS] = True

    iteration = 1
    while not all(converged) and iteration < maxiter:
        # 1. Get all non-converged timeseries
        y_sub = y[:, ~converged]
        _beta = beta.copy()

        # 2. Calculate new weights and do a weighted fit
        weights = M(resid[:, ~converged] / scale, c=tune)
        beta[:, ~converged], resid[:, ~converged] = weighted_ols(X, y_sub,
                                                                 weights)
        iteration += 1

        # 3. For all time series where the change in beta is smaller than the
        #   tolerance set convergence to True
        is_converged = ~np.any(np.fabs(beta - _beta > tol), axis=0)
        converged[is_converged] = True

        # If chosen repeat initialization by recalculating the scale
        if update_scale:
            est = scale_est(resid[:, ~converged], c=scale_constant)
            scale = np.where(EPS > est, EPS, est)

    if is_1d:
        resid = resid.squeeze(axis=-1)
        beta = beta.squeeze(axis=-1)

    return beta, resid


def weighted_ols(X, y, w):
    """Apply a weighted OLS fit to data

    Args:
        X (np.ndarray): independent variables
        y (np.ndarray): dependent variable
        w (np.ndarray): observation weights

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


def ccdc_stable_fit(X, y, dates, threshold=3, **kwargs):
    """Fitting stable regressions using an adapted CCDC method

    Models are first fit using OLS regression. Those models are then checked for
    stability with 'is_stable_ccdc()'. If a model is not stable, the two oldest
    acquisitions are removed, a model is fit using this shorter
    time-series and again checked for stability. This process continues as long
    as all of the following 3 conditions are met:

    1. There are unstable timeseries left.
    2. There are enough cloud-free acquisitions left (threshold is 1.5x the
        number of parameters in the design matrix).
    3. There is still data of more than 1 year available.

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        dates ((M, ) np.ndarray): Corresponding dates to y in numpy datetime64
        threshold (float): Sensitivity of stability checking. Gets passed to
            ``is_stable_ccdc()``
    Returns:
        beta (numpy.ndarray): The array of regression estimators
        residuals (numpy.ndarray): The array of residuals
        is_stable (numpy.ndarray): 1D Boolean array indicating stability
    """
    # 0. Remove observations with too little data
    # Minimum 1.5 times the number of coefficients
    obs_count = np.count_nonzero(~np.isnan(y), axis=0)
    enough = obs_count > X.shape[1] * 1.5
    is_stable = np.full(enough.shape, False, dtype=np.bool)
    y_sub = y[:, enough]
    X_sub = X

    # Initialize dates to check if there's acquisitions for an entire year
    first_date = dates[0]
    last_date = dates[-1]
    delta = last_date - first_date

    # If the dates are less than one year apart Raise an Exception
    if delta.astype('timedelta64[Y]') < np.timedelta64(1, 'Y'):
        raise ValueError('"dates" requires a full year of data.')

    # Initialize beta and residuals filled with nan
    beta = np.full([X.shape[1], y.shape[1]], np.nan, dtype=np.float32)
    residuals = np.full(y.shape, np.nan, dtype=np.float32)

    # Keep going while everything isn't either stable or has enough data left
    while not np.all(is_stable | ~enough):
        # 1. Fit
        beta_sub, residuals_sub = ols(X_sub, y_sub)
        beta[:,~is_stable & enough] = beta_sub
        residuals[:,~is_stable & enough] = np.nan
        residuals[-y_sub.shape[0]:,~is_stable & enough] = residuals_sub

        # 2. Check stability
        is_stable_sub = is_stable_ccdc(beta_sub[1, :], residuals_sub, threshold)

        # 3. Update mask
        # Everything that wasn't stable last time and had enough data gets updated
        is_stable[~is_stable & enough] = is_stable_sub

        # 4. Change Timeframe and remove everything that is now stable
        y_sub = y_sub[2:,~is_stable_sub]
        X_sub = X_sub[2:,:]
        logger.debug('Fitted %d stable pixels.',
                     is_stable_sub.shape[0]-y_sub.shape[1])
        dates = dates[2:]
        first_date = dates[0]
        delta = last_date - first_date

        # If the dates are less than one year apart stop the loop
        if delta.astype('timedelta64[Y]') < np.timedelta64(1, 'Y'):
            break
        # Check where there isn't enough data left
        obs_count = np.count_nonzero(~np.isnan(y_sub), axis=0)
        enough_sub = obs_count > X.shape[1] * 1.5
        enough[~is_stable & enough] = enough_sub
        # Remove everything where there isn't enough data
        y_sub = y_sub[:,enough_sub]
    return beta, residuals, is_stable


@numba.jit(nopython=True)
def roc_stable_fit(X, y, alpha=0.05):
    is_stable = np.ones(y.shape[1])
    beta = np.empty([X.shape[1], y.shape[1]], dtype=np.double)
    beta.fill(np.nan)
    for idx in range(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:, idx])
        _y = y[~is_nan, idx]
        _X = X[~is_nan, :]

        # get the index where the stable period starts
        stable_idx = history_roc(_X, _y, alpha=alpha)

        # If there are not enough observations available in the stable period
        # set stability to False and continue
        # TODO: Maybe also check if more than 1 year of data is available
        if _y.shape[0] - stable_idx < X.shape[1]*1.5:
            is_stable[idx] = 0
            continue

        # Subset and fit
        X_stable = _X[stable_idx:]
        y_stable = _y[stable_idx:]
        XTX = np.linalg.inv(np.dot(X_stable.T, X_stable))
        XTY = np.dot(X_stable.T, y_stable)
        beta[:, idx] = np.dot(XTX, XTY)

    residuals = np.dot(X, beta) - y
    return beta, residuals, is_stable.astype(np.bool)

