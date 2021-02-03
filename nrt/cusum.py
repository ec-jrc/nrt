import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

pnorm = norm.cdf


def history_roc(X, y, alpha=0.05):
    # Index, where unstability in time-series is detected
    #  0: time-series completely stable
    # >0: stable after this index
    unstable_idx = np.zeros(y.shape[1])
    for idx in range(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:, idx])
        _y = y[~is_nan, idx]
        _X = X[~is_nan, :]

        process = _cusum_rec_efp(_X[::-1], _y[::-1])
        stat = _cusum_rec_sctest(process)
        stat_pvalue = _brownian_motion_pvalue(stat, 1)
        if stat_pvalue < alpha:
            boundary = _cusum_rec_boundary(process, alpha)
            unstable_idx[idx] = len(process) \
                                - np.where(np.abs(process) > boundary)[0].min()
            # convert to correct index in the full timeseries with nan
            #unstable_idx[idx] = np.where(~is_nan)[0][unstable_sub]
        else:
            unstable_idx[idx] = 0
    return unstable_idx


# REC-CUSUM
def _brownian_motion_pvalue(x, k):
    """ Return pvalue for some given test statistic """
    # TODO: Make generic, add "type='Brownian Motion'"?
    if x < 0.3:
        p = 1 - 0.1464 * x
    else:
        p = 2 * (1 -
                 pnorm(3 * x) +
                 np.exp(-4 * x ** 2) * (pnorm(x) + pnorm(5 * x) - 1) -
                 np.exp(-16 * x ** 2) * (1 - pnorm(x)))
    return 1 - (1 - p) ** k


def _cusum_rec_boundary(x, alpha=0.05):
    """ Equivalent to ``strucchange::boundary.efp``` for Rec-CUSUM """
    n = x.ravel().size
    bound = _cusum_rec_test_crit(alpha)
    boundary = (bound + (2 * bound * np.arange(0, n) / (n - 1)))

    return boundary


def _cusum_rec_test_crit(alpha):
    """ Return critical test statistic value for some alpha """
    return brentq(lambda _x: _brownian_motion_pvalue(_x, 1) - alpha, 0, 20)


def _cusum_rec_efp(X, y):
    """ Equivalent to ``strucchange::efp`` for Rec-CUSUM """
    # Run "efp"
    n, k = X.shape
    k = k+1
    w = _recresid(X, y, k)[k:]
    sigma = np.std(w)# ** 0.5
    w = np.concatenate((np.array([0]), w))
    return np.cumsum(w) / (sigma * (n - k) ** 0.5)


def _cusum_rec_sctest(x):
    """ Equivalent to ``strucchange::sctest`` for Rec-CUSUM """
    x = x[1:]
    j = np.linspace(0, 1, x.size + 1)[1:]
    x = x * 1 / (1 + 2 * j)
    stat = np.abs(x).max()

    return stat


def _recresid(X, y, span):
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
