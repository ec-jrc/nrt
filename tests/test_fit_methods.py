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
