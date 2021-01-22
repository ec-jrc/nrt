"""Model fitting

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return y with outliers set to np.nan.
These functions are meant to be called in ``nrt.BaseNrt._fit()``

Citations:

- Brooks, E.B., Wynne, R.H., Thomas, V.A., Blinn, C.E. and Coulston, J.W., 2013.
  On-the-fly massively multitemporal change detection using statistical quality
  control charts and Landsat data. IEEE Transactions on Geoscience and Remote Sensing,
  52(6), pp.3316-3332.
- ADD ZHU
"""


def shewhart(X, y, L):
    """Remove outliers using a Shewhart control chart

    As described in Brooks et al. 2014, following an initial OLS fit, outliers are
    identified using a shewhart control chart and removed.

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ({(M,), (M, K)} np.ndarray): Matrix of dependant variables
        L (float): control limit used for outlier filtering. Must be a positive
            float. Lower values indicate stricter filtering

    Returns:
        y(np.ndarray): Dependant variables with outliers set to np.nan
    """
    beta_full, residuals_full = ols(X, y)
    # Shewhart chart to get rid of outliers (clouds etc)
    sigma = np.nanstd(residuals_full, axis=0)
    shewhart_mask = np.abs(residuals_full) > L * sigma
    y[shewhart_mask] = np.nan
    return y