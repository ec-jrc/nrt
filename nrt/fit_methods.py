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

from nrt.stats import nanlstsq


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
