import numba
import numpy as np


#@numba.jit(nopython=True)
def nanlstsq(X, y):
    """Return the least-squares solution to a linear matrix equation

    Analog to ``numpy.linalg.lstsq`` for dependant variable containing ``Nan``

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ({(M,), (M, K)} np.ndarray): Matrix of dependant variables

    Examples:
        >>> import numpy as np
        >>> from sklearn.datasets import make_regression
        >>> from nrt.stats import nanlstsq
        >>> # Generate random data
        >>> n_targets = 1000
        >>> n_features = 2
        >>> X, y = make_regression(n_samples=200, n_features=n_features,
        ...                        n_targets=n_targets)
        >>> # Add random nan to y array
        >>> y.ravel()[np.random.choice(y.size, 5*n_targets, replace=False)] = np.nan
        >>> # Run the regression
        >>> beta = nanlstsq(X, y)
        >>> assert beta.shape == (n_features, n_targets)

    Returns:
        np.ndarray: Least-squares solution, ignoring ``Nan``
    """
    beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    for idx in range(y.shape[1]):
        isna = np.isnan(y[:,idx])
        X_sub = X[~isna]
        y_sub = y[~isna,idx]

        XTX = np.linalg.inv(np.dot(X_sub.T, X_sub))
        XTY = np.dot(X_sub.T, y_sub)
        beta[:,idx] = np.dot(XTX, XTY)

    return beta


#@numba.jit(nopython=True)
# axis prevents numba -> for loop might be faster after all
def mad(resid, c=0.6745):
    """
    Returns Median-Absolute-Deviation (MAD) for residuals
    Args:
        resid (np.ndarray): residuals
        c (float): scale factor to get to ~standard normal (default: 0.6745)
                 (i.e. 1 / 0.75iCDF ~= 1.4826 = 1 / 0.6745)
    Returns:
        float: MAD 'robust' variance estimate
    Reference:
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    # Return median absolute deviation adjusted sigma
    return np.nanmedian(
        np.fabs(resid - np.nanmedian(resid, axis=0)), axis=0) / c

# Weight scaling methods
@numba.jit(nopython=True)
def bisquare(resid, c=4.685):
    """
    Returns weighting for each residual using bisquare weight function
    Args:
        resid (np.ndarray): residuals to be weighted
        c (float): tuning constant for Tukey's Biweight (default: 4.685)
    Returns:
        weight (ndarray): weights for residuals
    Reference:
        http://statsmodels.sourceforge.net/stable/generated/statsmodels.robust.norms.TukeyBiweight.html
    """
    # Weight where abs(resid) < c; otherwise 0
    return (np.abs(resid) < c) * (1 - (resid / c) ** 2) ** 2

