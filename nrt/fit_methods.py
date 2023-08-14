"""Model fitting

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return both coefficient (beta) and residuals matrices.
These functions are meant to be called in ``nrt.BaseNrt._fit()``.

The RIRLS fit is derived from Chris Holden's yatsm package. See the
copyright statement below.
"""
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

###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2014 Chris Holden
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import numpy as np
import numba

from nrt.log import logger
from nrt import utils
from nrt.utils_efp import history_roc
from nrt.stats import nanlstsq, mad, bisquare


def ols(X, y):
    """Fit simple OLS model

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


@utils.numba_kwargs
@numba.jit(nopython=True, cache=True, parallel=True)
def rirls(X, y, M=bisquare, tune=4.685,
          scale_est=mad, scale_constant=0.6745,
          update_scale=True, maxiter=50, tol=1e-8):
    """Robust Linear Model using Iterative Reweighted Least Squares (RIRLS)

    Perform robust fitting regression via iteratively reweighted least squares
    according to weight function and tuning parameter.
    Basically a clone from `statsmodels` that should be much faster.

    Note:
        For best performances of the multithreaded implementation, it is
        recommended to limit the number of threads used by MKL or OpenBLAS to 1.
        This avoids over-subscription, and improves performances.
        By default the function will use all cores available; the number of cores
        used can be controled using the ``numba.set_num_threads`` function or
        by modifying the ``NUMBA_NUM_THREADS`` environment variable

    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D independent variable
        tune (float): tuning constant for scale estimate
        maxiter (int, optional): maximum number of iterations (default: 50)
        tol (float, optional): convergence tolerance of estimate
            (default: 1e-8)
        scale_est (callable): estimate used to scale the weights
            (default: `mad` for median absolute deviation)
        scale_constant (float): normalization constant (default: 0.6745)
        update_scale (bool, optional): update scale estimate for weights
            across iterations (default: True)
        M (callable): function for scaling residuals
        tune (float): tuning constant for scale estimate

    Returns:
        tuple: beta-coefficients and residual vector
    """
    beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    resid = np.full_like(y, np.nan, dtype=np.float64)
    for idx in numba.prange(y.shape[1]):
        y_sub = y[:,idx]
        isna = np.isnan(y_sub)
        X_sub = X[~isna]
        y_sub = y_sub[~isna]
        beta_, resid_ = weighted_ols(X_sub, y_sub, np.ones_like(y_sub))
        scale = scale_est(resid_, c=scale_constant)

        EPS = np.finfo(np.float32).eps
        if scale < EPS:
            beta[:,idx] = beta_
            resid[~isna,idx] = resid_
            continue

        iteration = 1
        converged = 0
        while not converged and iteration < maxiter:
            last_beta = beta_.copy()
            weights = M(resid_ / scale, c=tune)
            beta_, resid_ = weighted_ols(X_sub, y_sub, weights)
            if update_scale:
                scale = max(EPS,scale_est(resid_, c=scale_constant))
            iteration += 1
            converged = not np.any(np.fabs(beta_ - last_beta > tol))
        beta[:,idx] = beta_
        resid[~isna,idx] = resid_

    return beta, resid


@numba.jit(nopython=True, cache=True)
def weighted_ols(X, y, w):
    """Apply a weighted OLS fit to 1D data

    Args:
        X (np.ndarray): independent variables
        y (np.ndarray): dependent variable
        w (np.ndarray): observation weights

    Returns:
        tuple: coefficients and residual vector
    """
    sw = np.sqrt(w)

    Xw = X * np.expand_dims(sw, -1)
    yw = y * sw

    beta,_,_,_ = np.linalg.lstsq(Xw, yw)

    resid = y - np.dot(X, beta)

    return beta, resid

@utils.numba_kwargs
@numba.jit(nopython=True, cache=True, parallel=True)
def ccdc_stable_fit(X, y, dates, threshold=3):
    """Fitting stable regressions using an adapted CCDC method

    Models are first fit using OLS regression. Those models are then checked for
    stability. If a model is not stable, the two oldest
    acquisitions are removed, a model is fit using this shorter
    time-series and again checked for stability. This process continues as long
    as all of the following 3 conditions are met:

    1. The timeseries is still unstable
    2. There are enough cloud-free acquisitions left (threshold is 1.5x the
        number of parameters in the design matrix)
    3. The time series includes data of more than half a year

    Stability depends on all these three conditions being true:
    1.             slope / RMSE < threshold
    2. first observation / RMSE < threshold
    3.  last observation / RMSE < threshold

    Note:
        For best performances of the multithreaded implementation, it is
        recommended to limit the number of threads used by MKL or OpenBLAS to 1.
        This avoids over-subscription, and improves performances.
        By default the function will use all cores available; the number of cores
        used can be controled using the ``numba.set_num_threads`` function or
        by modifying the ``NUMBA_NUM_THREADS`` environment variable

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        dates ((M, ) np.ndarray): Corresponding dates to y in numpy datetime64
        threshold (float): Sensitivity of stability checking. Gets passed to
            ``is_stable_ccdc()``
    Returns:
        beta (numpy.ndarray): The array of regression estimators
        residuals (numpy.ndarray): The array of residuals
        is_stable (numpy.ndarray): 1D Boolean array indicating stability
        start (numpy.ndarray): 1D integer array indicating day of fitting start
            as days since UNIX epoch.
    """
    min_obs = int(X.shape[1] * 1.5)
    beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    residuals = np.full_like(y, np.nan)
    stable = np.empty((y.shape[1]))
    fit_start = np.empty((y.shape[1]))
    for idx in numba.prange(y.shape[1]):
        y_sub = y[:, idx]
        isna = np.isnan(y_sub)
        X_sub = X[~isna]
        y_sub = y_sub[~isna]
        _dates = dates[~isna]
        is_stable = False

        # Run until minimum observations
        # or until stability is reached
        for jdx in range(len(y_sub), min_obs-1, -2):
            # Timeseries gets reduced by two elements
            # each iteration
            y_ = y_sub[-jdx:]
            X_ = X_sub[-jdx:]
            beta_sub = np.linalg.solve(np.dot(X_.T, X_), np.dot(X_.T, y_))
            resid_sub = np.dot(X_, beta_sub) - y_

            # Check for stability
            rmse = np.sqrt(np.mean(resid_sub ** 2))
            slope = np.fabs(beta_sub[1]) / rmse < threshold
            first = np.fabs(resid_sub[0]) / rmse < threshold
            last = np.fabs(resid_sub[-1]) / rmse < threshold

            # Break if stability is reached
            is_stable = slope & first & last
            if is_stable:
                break
            # Also break if less than half a year of data remain
            last_date = _dates[-1]
            first_date = _dates[-jdx]
            if last_date - first_date < 183:
                break

        beta[:,idx] = beta_sub
        residuals[-jdx:,idx] = resid_sub
        stable[idx] = is_stable
        fit_start[idx] = _dates[-jdx]
    return beta, residuals, stable.astype(np.bool_), fit_start


@utils.numba_kwargs
@numba.jit(nopython=True, cache=True, parallel=False)
def roc_stable_fit(X, y, dates, alpha=0.05, crit=0.9478982340418134):
    """Fitting stable regressions using Reverse Ordered Cumulative Sums

    Calculates OLS coefficients, residuals and a stability mask based on
    a stable history period which is provided by ``history_roc()``.

    The pixel will get marked as unstable if:
    1. The stable period is shorter than half a year OR
    2. There are fewer observation than the number of coefficients in X

    The implementation roughly corresponds to the fit of bfastmonitor
    with the history option set to 'ROC'.

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        dates ((M, ) np.ndarray): Corresponding dates to y in days since epoch
            (int)
        alpha (float): Significance level for the boundary
            (probability of type I error)
        crit (float): Critical value corresponding to the chosen alpha. Can be
            calculated with ``_cusum_rec_test_crit``.
            Default is the value for alpha=0.05

    Returns:
        beta (numpy.ndarray): The array of regression estimators
        residuals (numpy.ndarray): The array of residuals
        is_stable (numpy.ndarray): 1D Boolean array indicating stability
        start (numpy.ndarray): 1D integer array indicating day of fitting start
            as days since UNIX epoch.
    """
    is_stable = np.ones(y.shape[1], dtype=np.bool_)
    fit_start = np.zeros_like(is_stable, dtype=np.uint16)
    beta = np.full((X.shape[1], y.shape[1]), np.nan, dtype=np.float64)
    nreg = X.shape[1]
    for idx in numba.prange(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:, idx])
        _y = y[~is_nan, idx]
        _X = X[~is_nan, :]

        # get the index where the stable period starts
        stable_idx = history_roc(_X, _y, alpha=alpha, crit=crit)

        # If there are not enough observations available in the stable period
        # set stability to False and continue
        if len(_y) - stable_idx < nreg + 1:
            is_stable[idx] = False
            continue

        # Check if there is more than 1 year (365 days) of data available
        # If not, set stability to False and continue
        _dates = dates[~is_nan]
        last_date = _dates[-1]
        first_date = _dates[stable_idx]
        if last_date - first_date < 183:
            is_stable[idx] = False
            continue

        # Subset and fit
        X_stable = _X[stable_idx:]
        y_stable = _y[stable_idx:]
        beta[:, idx] = np.linalg.solve(np.dot(X_stable.T, X_stable),
                                       np.dot(X_stable.T, y_stable))
        fit_start[idx] = _dates[stable_idx]

    residuals = np.dot(X, beta) - y
    return beta, residuals, is_stable, fit_start
