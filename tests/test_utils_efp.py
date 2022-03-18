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

import numpy as np
import xarray as xr
import pytest
import nrt.utils_efp as cs
from nrt.monitor.cusum import CuSum
from nrt.monitor.mosum import MoSum


def test_history_roc(X_y_dates_romania):
    """Validated against roc_history() in the R package bFast
    res_2d <- apply(y, 2, function(column){
      # convert to dataframe
      X_df <- as.data.frame(X)
      X_df$y <- column
      #Remove nan
      X_df_clear <- X_df[!is.na(X_df$y),]

      level <- 0.05

      n <- nrow(X_df_clear)
      data_rev <- X_df_clear[n:1, ]
      y_rcus <- efp(y ~ V1+V2+V3+V4+V5, data = data_rev, type = "Rec-CUSUM")
      y_start <- if (sctest(y_rcus)$p.value < level) {
        length(y_rcus$process) - min(which(abs(y_rcus$process)[-1] >
                                             boundary(y_rcus)[-1])) + 1
      } else {
        1
      }
      return(y_start)
    })
    """
    X, y, dates = X_y_dates_romania
    result = np.array([1, 8, 49, 62, 1], dtype='float32')
    stable_idx = np.zeros(y.shape[1])
    for idx in range(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:, idx])
        _y = y[~is_nan, idx]
        _X = X[~is_nan, :]

        # get the index where the stable period starts
        stable_idx[idx] = cs.history_roc(_X, _y)

    # Result from strucchange must be subtracted by 1, because R is 1 indexed
    np.testing.assert_allclose(stable_idx, result-1)


def test_efp(X_y_dates_romania, strcchng_efp):
    """Test efp against process value of
    strucchange::efp with type='Rec-CUSUM'"""
    X, y, dates = X_y_dates_romania

    is_nan = np.isnan(y[:, 0])
    _y = y[~is_nan, 0]
    _X = X[~is_nan, :]

    process = cs._cusum_rec_efp(_X[::-1], _y[::-1])

    result = strcchng_efp

    # Relative high tolerance, due to floating point precision
    np.testing.assert_allclose(process[X.shape[1]+2:], result[X.shape[1]+2:],
                               rtol=1e-02)


@pytest.mark.parametrize("test_input,expected", [(0.01, 3.368214),
                                                 (0.05, 2.795483),
                                                 (0.1, 2.500278)])
def test_cusum_ols_test_crit(test_input, expected):
    assert cs._cusum_ols_test_crit(test_input) == pytest.approx(expected)


mosum_crit_params = {
    'h': (pytest.raises(ValueError), {'alpha': 0.05, 'h': 0.24}),
    'alpha': (pytest.raises(ValueError), {'alpha': 0.06}),
    'period': (pytest.raises(ValueError), {'alpha': 0.05, 'period': 11}),
}

@pytest.mark.parametrize('expected, test_input', mosum_crit_params.values(),
                         ids=mosum_crit_params.keys())
def test_mosum_ols_test_crit(expected, test_input):
    """Test edge cases"""
    with expected:
        assert cs._mosum_ols_test_crit(**test_input) is not None


def test_process_boundary_cusum(X_y_dates_romania, cusum_result):
    X, y, dates = X_y_dates_romania
    # make y 6 long
    y = np.insert(y, 5, values=y[:,0], axis=1)
    y_3d = y.reshape((y.shape[0], 2, -1))
    data = xr.DataArray(y_3d, dims=["time", "x", "y"], coords={"time": dates})
    fit = data[:100]
    monitor = data[100:]
    cusum_monitor = CuSum(trend=False)
    cusum_monitor.fit(dataarray=fit, method='OLS')
    for array, date in zip(monitor.values,
                           monitor.time.values.astype('datetime64[s]').tolist()):
        cusum_monitor.monitor(array=array, date=date)

    # Process value
    np.testing.assert_allclose(cusum_result[0],
                               cusum_monitor.process.ravel()[:-1], rtol=1e-4)
    # Boundary value
    np.testing.assert_allclose(cusum_result[1],
                               cusum_monitor.boundary.ravel()[:-1])
    # Histsize
    np.testing.assert_allclose(cusum_result[2],
                               cusum_monitor.histsize.ravel()[:-1])
    # Sigma
    np.testing.assert_allclose(cusum_result[3],
                               cusum_monitor.sigma.ravel()[:-1], rtol=1e-6)


def test_process_boundary_mosum(X_y_dates_romania, mosum_result):
    X, y, dates = X_y_dates_romania
    # make y 6 long
    y = np.insert(y, 5, values=y[:,0], axis=1)
    y_3d = y.reshape((y.shape[0], 2, -1))
    data = xr.DataArray(y_3d, dims=["time", "x", "y"], coords={"time": dates})
    fit = data[:100]
    monitor = data[100:]
    mosum_monitor = MoSum(trend=False)
    mosum_monitor.fit(dataarray=fit, method='OLS')
    for array, date in zip(monitor.values,
                           monitor.time.values.astype('datetime64[s]').tolist()):
        mosum_monitor.monitor(array=array, date=date)

    # Process value (third value has a break and so diverges a lot since
    # monitoring in bFast does not stop in case there is a break)
    np.testing.assert_allclose(np.delete(mosum_result[0], 2),
                               np.delete(mosum_monitor.process.ravel(), [2,-1]),
                               rtol=1e-4)
    # Boundary value
    np.testing.assert_allclose(mosum_result[1],
                               mosum_monitor.boundary.ravel()[:-1])
    # Histsize
    np.testing.assert_allclose(mosum_result[2],
                               mosum_monitor.histsize.ravel()[:-1])
    # Sigma
    np.testing.assert_allclose(mosum_result[3],
                               mosum_monitor.sigma.ravel()[:-1], rtol=1e-6)
