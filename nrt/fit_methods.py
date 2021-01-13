import numpy as np
from nrt.stats import nanlstsq


def shewhart(X, dataarray, threshold, **kwargs):
    beta_full, residuals_full = ols(X, dataarray=dataarray)

    # Shewhart chart to get rid of outliers (clouds etc)
    sigma = np.nanstd(residuals_full, axis=0)
    shewhart_mask = np.abs(residuals_full) > (threshold * sigma)
    dataarray.values[shewhart_mask] = np.nan

    # fit again, but without outliers
    beta, residuals = ols(X, dataarray=dataarray)
    return beta, residuals


def ols(X, dataarray, **kwargs):
    y = dataarray.values.astype(np.float32)
    X = X.astype(np.float32)
    shape = y.shape
    shape_flat = (shape[0], shape[1] * shape[2])
    beta_shape = (X.shape[1], shape[1], shape[2])
    y_flat = y.reshape(shape_flat)
    beta = nanlstsq(X, y_flat)
    residuals = np.dot(X, beta) - y_flat
    return beta.reshape(beta_shape), residuals.reshape(shape)
