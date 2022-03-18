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

import nrt.fit_methods as fm
import nrt.stats as st


def test_rirls(X_y_RLM, sm_RLM_result):
    """
    Compare against implementation in statsmodels
    """
    X, y = X_y_RLM
    beta, residuals = fm.rirls(X, y, M=st.bisquare, tune=4.685,
               scale_est=st.mad, scale_constant=0.6745, update_scale=True,
               maxiter=50, tol=1e-8)

    np.testing.assert_allclose(beta, sm_RLM_result, rtol=1e-02)


def test_roc_stable_fit(X_y_dates_romania):
    """Only integration test"""
    X, y, dates = X_y_dates_romania
    dates = dates.astype('datetime64[D]').astype('int')
    beta, resid, is_stable, fit_start = fm.roc_stable_fit(X.astype(np.float64),
                                                          y.astype(np.float64),
                                                          dates)

def test_ccdc_stable_fit(stability_ccdc, threshold=3):
    X, y, dates, result = stability_ccdc
    beta, resid, stable, start = fm.ccdc_stable_fit(X, y, dates, threshold)
    np.testing.assert_array_equal(result, stable)
