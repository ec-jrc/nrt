# Copyright (C) 2022 European Union (Joint Research Centre)
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#   https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

import numba
import numpy as np


@numba.jit(nopython=True, cache=True, parallel=True)
def nanlstsq(X, y):
    """Return the least-squares solution to a linear matrix equation

    Analog to ``numpy.linalg.lstsq`` for dependant variable containing ``Nan``

    Note:
        For best performances of the multithreaded implementation, it is
        recommended to limit the number of threads used by MKL or OpenBLAS to 1.
        This avoids over-subscription, and improves performances.
        By default the function will use all cores available; the number of cores
        used can be controled using the ``numba.set_num_threads`` function or
        by modifying the ``NUMBA_NUM_THREADS`` environment variable

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ({(M,), (M, K)} np.ndarray): Matrix of dependant variables

    Examples:
        >>> import os
        >>> # Adjust linear algebra configuration (only one should be required
        >>> # depending on how numpy was installed/compiled)
        >>> os.environ['OPENBLAS_NUM_THREADS'] = '1'
        >>> os.environ['MKL_NUM_THREADS'] = '1'
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
    for idx in numba.prange(y.shape[1]):
        # subset y and X
        isna = np.isnan(y[:,idx])
        X_sub = X[~isna]
        y_sub = y[~isna,idx]
        beta[:, idx] = np.linalg.solve(np.dot(X_sub.T, X_sub), np.dot(X_sub.T, y_sub))
    return beta


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
    return np.nanmedian(np.fabs(resid - np.nanmedian(resid))) / c

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


@numba.jit(nopython=True, cache=True)
def nan_percentile_axis0(arr, percentiles):
    """Faster implementation of np.nanpercentile

    This implementation always takes the percentile along axis 0.
    Uses numba to speed up the calculation by more than 7x.

    Function is equivalent to np.nanpercentile(arr, <percentiles>, axis=0)

    Args:
        arr (np.ndarray): 2D array to calculate percentiles for
        percentiles (np.ndarray): 1D array of percentiles to calculate

    Returns:
        np.ndarray: Array with first dimension corresponding to values passed
        in percentiles

    """
    shape = arr.shape
    arr = arr.reshape((arr.shape[0], -1))
    out = np.empty((len(percentiles), arr.shape[1]))
    for i in range(arr.shape[1]):
        out[:,i] = np.nanpercentile(arr[:,i], percentiles)
    shape = (out.shape[0], *shape[1:])
    return out.reshape(shape)
