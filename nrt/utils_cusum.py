"""CUSUM utility functions

Functions defined in this module implement functionality necessary for
CUSUM and MOSUM monitoring as implemented in the R packages strucchange and
bFast.

Portions of this module are derived from Chris Holden's pybreakpoints package.
See the copyright statement below.
"""

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, Chris Holden
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
import numba
from scipy import optimize
from scipy.stats import norm

from nrt.stats import ncdf
from nrt import data


@numba.jit(nopython=True)
def history_roc(X, y, alpha=0.05, crit=0.9478982340418134):
    """Reverse Ordered Rec-CUSUM check for stable periods

    Checks for stable periods by calculating recursive OLS-Residuals
    (see ``_recresid()``) on the reversed X and y matrices. If the cumulative
    sum of the residuals crosses a boundary, the index of y where this
    structural change occured is returned.

    Args:
        X ((M, ) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        alpha (float): Significance level for the boundary
            (probability of type I error)
        crit (float): Critical value corresponding to the chosen alpha. Can be
            calculated with ``_cusum_rec_test_crit``.
            Default is the value for alpha=0.05
    Returns:
        (int) Index of structural change in y.
             0: y completely stable
            >0: y stable after this index

    """
    # Index, where instability in time-series is detected
    #  0: time-series completely stable
    # >0: stable after this index
    process = _cusum_rec_efp(X[::-1], y[::-1])
    stat = _cusum_rec_sctest(process)
    stat_pvalue = _brownian_motion_pvalue(stat, 1)
    if stat_pvalue < alpha:
        boundary = _cusum_rec_boundary(process, crit)
        return len(process) - np.where(np.abs(process) > boundary)[0].min()
    else:
        return 0


# REC-CUSUM
@numba.jit(nopython=True)
def _brownian_motion_pvalue(x, k):
    """ Return pvalue for some given test statistic """
    # TODO: Make generic, add "type='Brownian Motion'"?
    if x < 0.3:
        p = 1 - 0.1464 * x
    else:
        p = 2 * (1 -
                 ncdf(3 * x) +
                 np.exp(-4 * x ** 2) * (ncdf(x) + ncdf(5 * x) - 1) -
                 np.exp(-16 * x ** 2) * (1 - ncdf(x)))
    return 1 - (1 - p) ** k


@numba.jit(nopython=True)
def _cusum_rec_boundary(x, crit=0.9478982340418134):
    """ Equivalent to ``strucchange::boundary.efp``` for Rec-CUSUM

    Args:
        x (np.ndarray): Process values
        crit (float): Critical value as computed by _cusum_rec_test_crit.
            Default is the value for alpha=0.05
    """
    n = x.size
    bound = crit
    boundary = (bound + (2 * bound * np.arange(0, n) / (n - 1)))

    return boundary


def _cusum_rec_test_crit(alpha):
    """ Return critical test statistic value for some alpha """
    return optimize.brentq(lambda _x: _brownian_motion_pvalue(_x, 1) - alpha, 0, 20)


def _cusum_ols_test_crit(alpha):
    """ Return critical test statistic value for some alpha """
    return optimize.golden(lambda _x: np.abs(
        2 * (norm.cdf(_x) - _x * norm.pdf(_x)) + alpha - 2), brack=(0, 10))


def _mosum_ols_test_crit(alpha):
    sig_level, crit_dict = data.mre_crit_table()
    win_size = 0.5
    period = 10
    functional = "max"
    crit_values = crit_dict.get(str(win_size)) \
                           .get(str(period)) \
                           .get(functional)
    return np.interp(1 - alpha, sig_level, crit_values)


@numba.jit(nopython=True)
def _cusum_rec_efp(X, y):
    """ Equivalent to ``strucchange::efp`` for Rec-CUSUM """
    # Run "efp"
    n, k = X.shape
    k = k+1
    w = _recresid(X, y, k)[k:]
    sigma = np.std(w)
    w = np.concatenate((np.array([0]), w))
    return np.cumsum(w) / (sigma * (n - k) ** 0.5)


@numba.jit(nopython=True)
def _cusum_rec_sctest(x):
    """ Equivalent to ``strucchange::sctest`` for Rec-CUSUM """
    x = x[1:]
    j = np.linspace(0, 1, x.size + 1)[1:]
    x = x * 1 / (1 + 2 * j)
    stat = np.abs(x).max()

    return stat


@numba.jit(nopython=True)
def _recresid(X, y, span):
    """ Return standardized recursive residuals for y ~ X
    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        span (int): Minimum number of observations for initial regression.

    Returns:
        (np.ndarray) containing recursive residuals standardized by
            prediction error variance

    Notes:
        For a matrix :math:`X_t` of :math:`T` total observations of :math:`n`
        variables, the :math:`t` th recursive residual is the forecast prediction
        error for :math:`y_t` using a regression fit on the first :math:`t - 1`
        observations. Recursive residuals are scaled and standardized so they are
        :math:`N(0, 1)` distributed.
        Using notation from Brown, Durbin, and Evans (1975) and Judge, et al
        (1985):
        .. math::
            w_r =
                \\frac{y_r - \\boldsymbol{x}_r^{\prime}\\boldsymbol{b}_{r-1}}
                      {\sqrt{(1 + \\boldsymbol{x}_r^{\prime}
                       S_{r-1}\\boldsymbol{x}_r)}}
                =
                \\frac
                    {y_r - \\boldsymbol{x}_r^{\prime}\\boldsymbol{b}_r}
                    {\sqrt{1 - \\boldsymbol{x}_r^{\prime}S_r\\boldsymbol{x}_r}}
            r = k + 1, \ldots, T,
        where :math:`S_{r}` is the residual sum of squares after
        fitting the model on :math:`r` observations.
        A quick way of calculating :math:`\\boldsymbol{b}_r` and
        :math:`S_r` is using an update formula (Equations 4 and 5 in
        Brown, Durbin, and Evans; Equation 5.5.14 and 5.5.15 in Judge et al):
        .. math::
            \\boldsymbol{b}_r
                =
                b_{r-1} +
                \\frac
                    {S_{r-1}\\boldsymbol{x}_j
                        (y_r - \\boldsymbol{x}_r^{\prime}\\boldsymbol{b}_{r-1})}
                    {1 + \\boldsymbol{x}_r^{\prime}S_{r-1}x_r}
        .. math::
            S_r =
                S_{j-1} -
                \\frac{S_{j-1}\\boldsymbol{x}_r\\boldsymbol{x}_r^{\prime}S_{j-1}}
                      {1 + \\boldsymbol{x}_r^{\prime}S_{j-1}\\boldsymbol{x}_r}

    See Also:
        statsmodels.stats.diagnostic.recursive_olsresiduals
    """
    nobs, nvars = X.shape

    recresid_ = np.nan * np.zeros((nobs))
    recvar = np.nan * np.zeros((nobs))

    X0 = X[:span, :]
    y0 = y[:span]

    # Initial fit
    XTX_j = np.linalg.inv(np.dot(X0.T, X0))
    XTY = np.dot(X0.T, y0)
    beta = np.dot(XTX_j, XTY)

    yhat_j = np.dot(X[span - 1, :], beta)
    recresid_[span - 1] = y[span - 1] - yhat_j
    recvar[span - 1] = 1 + np.dot(X[span - 1, :],
                                  np.dot(XTX_j, X[span - 1, :]))
    for j in range(span, nobs):
        x_j = X[j:j+1, :]
        y_j = y[j]

        # Prediction with previous beta
        resid_j = y_j - np.dot(x_j, beta)

        # Update
        XTXx_j = np.dot(XTX_j, x_j.T)
        f_t = 1 + np.dot(x_j, XTXx_j)
        XTX_j = XTX_j - np.dot(XTXx_j, XTXx_j.T) / f_t  # eqn 5.5.15

        beta = beta + (XTXx_j * resid_j / f_t).ravel()  # eqn 5.5.14
        recresid_[j] = resid_j.item()
        recvar[j] = f_t.item()

    return recresid_ / np.sqrt(recvar)
