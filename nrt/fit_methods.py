import numpy as np
from nrt.stats import nanlstsq


def shewhart(X, y_flat, shape, threshold, **kwargs):
    beta_full, residuals_full = ols(X, y_flat)

    # Shewhart chart to get rid of outliers (clouds etc)
    sigma = np.nanstd(residuals_full.reshape(shape), axis=0)
    shewhart_mask = np.abs(residuals_full) > (threshold * sigma.reshape(y_flat.shape[1]))
    y_flat[shewhart_mask] = np.nan

    # fit again, but without outliers
    beta, residuals = ols(X, y_flat)
    return beta, residuals


def ols(X, y_flat, **kwargs):
    beta = nanlstsq(X, y_flat)
    residuals = np.dot(X, beta) - y_flat
    return beta, residuals
