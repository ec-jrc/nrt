import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
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
        # subset y and X
        isna = np.isnan(y[:,idx])
        X_sub = X[~isna]
        y_sub = y[~isna,idx]
        # Compute beta on data subset
        XTX = np.linalg.inv(np.dot(X_sub.T, X_sub))
        XTY = np.dot(X_sub.T, y_sub)
        beta[:,idx] = np.dot(XTX, XTY)
    return beta


@numba.jit(nopython=True, cache=True)
def weighted_nanlstsq(X, y):
    """Return the weighted least-squares solution to a linear matrix equation

    Analog to ``nanlstsq`` but X is unique for every observation in y

    Args:
        X ((K, M, N) np.ndarray): Weighted Matrix of independant variables,
            unique for each timeseries K in y
        y ((M, K) np.ndarray): Matrix of dependant variables

    Returns:
        np.ndarray: Least-squares solution for every y with unique X
    """
    beta = np.zeros((X.shape[2], y.shape[1]), dtype=np.float32)
    for idx in range(y.shape[1]):
        # Indexing one after the other, because Numba isn't supporting
        # multiple advanced indices
        y_ = y[:,idx]
        X_ = X[idx]
        isna = np.isnan(y_)
        X_sub = X_[~isna]
        y_sub = y_[~isna]

        XTX = np.linalg.inv(np.dot(X_sub.T, X_sub))
        XTY = np.dot(X_sub.T, y_sub)
        beta[:,idx] = np.dot(XTX, XTY)
    return beta


@numba.jit(nopython=True, cache=True)
def nanmedian_along_axis(arr, axis):
    """Returns Mean along selected axis

    Implementation to work with numba

    Args:
        arr (np.ndarray): N-Dimensional array
        axis (int): Axis to calculate the median along

    Returns:
        np.ndarray: Median excluding nan along the axis

    Reference:
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    res = np.empty(arr.shape[axis], dtype=arr.dtype)
    for idx in range(arr.shape[axis]):
        arr_sub = arr[:, idx]
        res[idx] = np.nanmedian(arr_sub)
    return res


@numba.jit(nopython=True, cache=True)
def mad(resid, c=0.6745):
    """Returns Median-Absolute-Deviation (MAD) for residuals

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
    return nanmedian_along_axis(
        np.fabs(resid - nanmedian_along_axis(resid, axis=1)), axis=1) / c

# Weight scaling methods
@numba.jit(nopython=True, cache=True)
def bisquare(resid, c=4.685):
    """Weight residuals using bisquare weight function

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


def is_stable_ccdc(slope, residuals, threshold):
    """Check the stability of the fitted model using CCDC Method

    Stability is given if:
        1.             slope / RMSE < threshold
        2. first observation / RMSE < threshold
        3.  last observation / RMSE < threshold

    For multiple bands Zhu et al. 2014 proposed the mean of all bands to
    be > 1 to signal instability.

    Args:
        slope (np.ndarray): 1D slope/trend of coefficients
        residuals (np.ndarray): 2D corresponding residuals
        threshold (float): threshold value to signal change

    Returns:
        np.ndarray: 1D boolean array with True = stable
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


@numba.jit(nopython=True, cache=True)
def erfcc(x):
    """Complementary error function."""
    z = np.abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if x >= 0.:
        return r
    else:
        return 2. - r


@numba.jit(nopython=True, cache=True)
def ncdf(x):
    """Normal cumulative distribution function
    Source: Stackoverflow Unknown,
    https://stackoverflow.com/a/809402/12819237"""
    return 1. - 0.5*erfcc(x/(2**0.5))
