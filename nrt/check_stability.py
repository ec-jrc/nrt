"""Fitting stable models

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return coefficient (beta), residuals matrices as well as
a boolean mask indicating which variables aren't stable.
These functions are meant to be called in ``nrt.BaseNrt._fit()``

Citations:

- ADD ZHU
"""
import numpy as np

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