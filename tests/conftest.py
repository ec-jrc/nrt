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
    a_len = 30
    # build an example, where one time series has a large last value,
    # one a large first value, one a large slope and one just random residuals
    residuals = (np.random.rand(a_len, 4) - 0.5)*2
    residuals[0, 0] = 100
    residuals[-1, 1] = 100

    ts = np.array([
        np.ones(a_len),
        np.ones(a_len),
        np.arange(a_len)*20+5, # Large slope
        np.ones(a_len)
    ]).T
    # add a np.nan in there:
    ts[int(a_len/2),3] = np.nan

    X = np.array([np.ones(a_len), np.arange(a_len)]).T
    y = ts+residuals
    dates = np.linspace(1, 365, a_len)
    result = np.array([True, False, False, True])
    return X, y, dates, result


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
    return np.loadtxt(here / 'data' / 'cusum_result.csv',
                      delimiter=',', dtype=np.float64)


# Same as cusum_result only with type="OLS-MOSUM"
@pytest.fixture
def mosum_result(request):
    return np.loadtxt(here / 'data' / 'mosum_result.csv',
                      delimiter=',', dtype=np.float64)

# Test data for robust fit.
#
# First time-series can become singular if accuracy isn't sufficient
@pytest.fixture
def X_y_RLM(request):
    X = np.loadtxt(here / 'data' / 'RLM_X.csv',
                      delimiter=',', dtype=np.float64)
    y = np.loadtxt(here / 'data' / 'RLM_y.csv',
                      delimiter=',', dtype=np.float64)
    return X, y

# Result of Robust Fit with statsmodels
#
# With X, y = X_y_RLM()
    # import statsmodels as sm
    #
    # for idx in range(y.shape[1]):
    #     y_sub = y[:, idx]
    #     isna = np.isnan(y_sub)
    #     X_sub = X[~isna]
    #     endog = y_sub[~isna]
    #     rlm_model = sm.RLM(endog, X_sub, M=sm.robust.norms.TukeyBiweight())
    #     rlm_results = rlm_model.fit(update_scale=True)
    #     beta[:,idx] = rlm_results.params
@pytest.fixture
def sm_RLM_result(request):
    return np.array([['2.3757569983999076', '-51.621207292381314'],
       ['1.5919053949452396e-05', '-0.00019788972214892546'],
       ['4.960483948314601', '-73.95341088849317'],
       ['4.0427485592574195', '-17.66452192456504'],
       ['1.0676653146683237', '0.579422996703399'],
       ['-0.7172424822211365', '-49.52111301879781'],
       ['1.2701246101474761', '-38.324020145702654'],
       ['1.1329168669944791', '-9.034638787625045']], dtype='<U32').astype(np.float64)
