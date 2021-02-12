from pathlib import Path
import pytest
import numpy as np

here = Path(__file__).parent

@pytest.fixture
def X_y_clear(X_y_intercept_slope):
    # adds an array indicating 'clear' pixels as True and outliers as False
    X, y, intercept, slope = X_y_intercept_slope
    clear = np.ones_like(y).astype('bool')
    clear[9, 0] = False
    clear[0, 1] = False
    return X, y, clear


@pytest.fixture
def X_y_intercept_slope(request):
    np.random.seed(0)
    slope, intercept = 2., 5.
    X = np.c_[np.ones(10), np.arange(10)]
    y = np.array([slope * X[:, 1] + intercept,
                  slope * X[:, 1] + intercept])
    # Add noise (X_y_clear depends on the same noise)
    y[0, 9] = 0
    y[1, 0] = 23
    return X, y.T, intercept, slope


# fixture of 2D residuals with extreme values start and end
# 1D slope with extreme value and corresponding results in stability
@pytest.fixture
def stability_ccdc(request):
    np.random.seed(0)
    # build an example, where one pixel has a large first value,
    # one a large last value one a large slope and one just random residuals
    residuals = np.random.rand(20, 4) - 0.5
    residuals[0, 0] = 100
    residuals[-1, 1] = 100
    slope = np.array([0, 0, 10, 0])
    stability = np.array([False, False, False, True])
    return residuals, slope, stability


@pytest.fixture
def X_y_dates_romania(request):
    # Imported as double, to match precision of R computation
    X = np.loadtxt(here / 'data' / 'X.csv', delimiter=',', dtype=np.float64)
    y = np.loadtxt(here / 'data' / 'y.csv', delimiter=',', dtype=np.float64)
    dates = np.genfromtxt(here / 'data' / 'dates.csv', delimiter=',') \
        .astype("datetime64[ns]")

    return X, y, dates


# results of calculating recursive residuals of X_y_dates_romania by
# strucchange package in R
# Recursive Residuals for entire matrix
# Code:
# res_2d < - apply(y, 2, function(column){
#     non_nan < - which(is.finite(column))
#     y_clear < - column[non_nan]
#     X_clear < - X[non_nan,]
#     rresid_na < - rep(NA, length(column))
#
#     rresid < - recresid(X_clear, y_clear)
#
#     rresid_na[non_nan[ncol(X_clear) + 1:length(rresid)]] < - rresid
#     return (rresid_na)
# })
@pytest.fixture
def strcchng_rr(request):
    return np.genfromtxt(here / 'data' / 'rr_result.csv',
                      delimiter=',', dtype=np.float64, missing_values='NA')


# Results of calculating Rec-CUSUM efp process value with efp() from package
# strucchange
#     X_df < - as.data.frame(X)
#     X_df$y < - y[,1]
#     # Remove nan
#     X_df_clear < - X_df[! is.na(X_df$y), ]
#
#     level < - 0.05
#
#     n < - nrow(X_df_clear)
#     data_rev < - X_df_clear[n:1, ]
#     y_rcus < - efp(y
#     ~ V1 + V2 + V3 + V4 + V5, data = data_rev, type = "Rec-CUSUM")
#     return (y_rcus$process)
@pytest.fixture
def strcchng_efp(request):
    return np.genfromtxt(here / 'data' / 'efp_result.csv',
                         delimiter=',', dtype=np.float64, missing_values='NA')


# Results of monitoring with strucchange
# res_bound_proc < - apply(y, 2, function(column)
# {
#     # convert to dataframe
#     X_df < - as.data.frame(X)
#     X_df$y < - column
#     # Split in history and monitor
#     history < - X_df[1:100, ]
#     # Remove nan
#     history_clear < - history[! is.na(history$y), ]
#     monitor_clear < - X_df[! is.na(X_df$y), ]
#
#     history_efp < - efp(y
#     ~ V2 + V3 + V4 + V5, data = history, type = "OLS-CUSUM")
#     history_mefp < - mefp(history_efp)
#     monitor_data < - monitor(history_mefp, data=monitor_clear)
#     plot(monitor_data)
#     return (c(monitor_process = as.numeric(tail(monitor_data$process, 1)),
#               boundary = history_mefp$border(nrow(monitor_clear)),
#               histsize = history_mefp$histsize,
#               sigma = history_efp$sigma))
# })
@pytest.fixture
def cusum_result(request):
    return np.loadtxt(here / 'data' / 'monitor_result.csv',
                      delimiter=',', dtype=np.float64)


# Same as cusum_result only with type="OLS-MOSUM"
@pytest.fixture
def mosum_result(request):
    return np.loadtxt(here / 'data' / 'mosum_result.csv',
                      delimiter=',', dtype=np.float64)
