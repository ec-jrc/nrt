"""Fitting stable models

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return coefficient (beta), residuals matrices as well as
a boolean mask indicating which variables aren't stable.
These functions are meant to be called in ``nrt.BaseNrt._fit()``

Citations:

- ADD ZHU
"""
import numpy as np
from nrt.fit_methods import ols


def ccdc_stable_fit(X, y, dates, threshold=3, **kwargs):
    # 0. Remove observations with too little data
    # Minimum 1.5 times the number of coefficients
    obs_count = np.count_nonzero(~np.isnan(y), axis=0)
    enough = obs_count > X.shape[1] * 1.5
    is_stable = np.full(enough.shape, False, dtype=np.bool)
    y_sub = y[:, enough]
    X_sub = X

    # Initialize beta and residuals filled with nan
    beta = np.full([X.shape[1], y.shape[1]], np.nan, dtype=np.float32)
    residuals = np.full(y.shape, np.nan, dtype=np.float32)


    # Keep going while everything isn't either stable or has enough data left
    while not np.all(is_stable | ~enough):
        # 1. Fit
        beta_sub, residuals_sub = ols(X_sub, y_sub)

        beta[:,~is_stable & enough] = beta_sub
        residuals[:,~is_stable & enough] = np.nan
        residuals[0:y_sub.shape[0],~is_stable & enough] = residuals_sub

        # 2. Check stability
        is_stable_sub = is_stable_ccdc(beta_sub[1, :], residuals_sub, threshold)

        # 3. Update mask
        # Everything that wasn't stable last time and had enough data gets updated
        is_stable[~is_stable & enough] = is_stable_sub
        print(np.count_nonzero(is_stable))

        # 4. Change Timeframe and remove everything that is now stable
        y_sub = y_sub[0:-2,~is_stable_sub]
        X_sub = X_sub[0:-2,:]

        # Then check where there isn't enough data left
        obs_count = np.count_nonzero(~np.isnan(y_sub), axis=0)
        enough_sub = obs_count > X.shape[1] * 1.5
        enough[~is_stable & enough] = enough_sub

        # Remove everything where there isn't enough data
        y_sub = y_sub[:,enough_sub]

        # # Re-fit
        # beta_sub, residuals_sub = ols(X_sub, y_sub)

    return beta, residuals, is_stable


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

    # It's only stable if all conditions are met
    is_stable = slope_rmse & first & last

    return is_stable
