import numba
import numpy as np


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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
@numba.jit(nopython=True)
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


def recresid(X, y, span=None):
    nvars = X.shape[1]
    if span is None:
        span = nvars

    # init result arrays
    norm_rresid = np.nan * np.zeros_like(y)

    for idx in range(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:,idx])
        y_sub = y[~is_nan,idx]
        y0 = y_sub[:span]
        X_sub = X[~is_nan,:]
        X0 = X_sub[:span]

        # set number of observations
        nobs = X_sub.shape[0]

        # init sub-result arrays
        recresid_sub = np.nan * np.zeros_like(y_sub)
        recvar_sub = np.nan * np.zeros_like(y_sub)

        # Initial fit
        XTX_j = np.linalg.inv(np.dot(X0.T, X0))
        XTY = np.dot(X0.T, y0)
        beta = np.dot(XTX_j, XTY)

        # First prediction
        yhat_j = np.dot(X_sub[span - 1], beta)
        recresid_sub[span - 1] = y_sub[span - 1] - yhat_j
        recvar_sub[span - 1] = 1 + np.dot(X_sub[span - 1],
                                      np.dot(XTX_j, X_sub[span - 1]))
        for j in range(span, nobs):
            x_j = X_sub[j:j + 1, :]
            y_j = y_sub[j]

            # Prediction with previous beta
            yhat_j = np.dot(x_j, beta)
            resid_j = y_j - yhat_j

            # Update
            XTXx_j = np.dot(XTX_j, x_j.T)
            f_t = 1 + np.dot(x_j, XTXx_j)
            XTX_j = XTX_j - np.dot(XTXx_j, XTXx_j.T) / f_t  # eqn 5.5.15

            beta = beta + (XTXx_j * resid_j / f_t).ravel()  # eqn 5.5.14

            recresid_sub[j] = resid_j
            recvar_sub[j] = f_t

        # Write sub result to full result array
        sigma = np.sqrt(recvar_sub)
        norm_rresid_sub = recresid_sub / sigma
        norm_rresid[~is_nan,idx] = norm_rresid_sub

    return norm_rresid


def history_roc(X, y, span, alpha=0.05):
    # Index, where unstability in time-series is detected
    #  0: time-series completely stable
    # >0: stable after this index
    unstable_idx = np.zeros(y.shape[1])
    for idx in range(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:, idx])
        _y = y[~is_nan, idx]
        _X = X[~is_nan, :]

        process = _cusum_rec_efp(_X, _y, span)
        stat = _cusum_rec_sctest(process)
        stat_pvalue = _brownian_motion_pvalue(stat, 1)
        if stat_pvalue < alpha:
            boundary = _cusum_rec_boundary(process, alpha)
            unstable_sub = len(process) \
                                - np.where(np.abs(process) > boundary)[0].min()
            # convert to correct index in the full timeseries with nan
            unstable_idx[idx] = np.where(~is_nan)[0][unstable_sub]
        else:
            unstable_idx[idx] = 0
    return unstable_idx