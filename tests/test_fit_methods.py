import numpy as np

import nrt.fit_methods as fm
import nrt.stats as st


def test_rirls(X_y_intercept_slope):
    """
    Compare against implementation in yatsm
    https://github.com/ceholden/yatsm/blob/
    8e328f366c8fd94d5cc57cd2cc42080c43d1f391/yatsm/regression/robust_fit.py
    """
    X, y, intercept, slope = X_y_intercept_slope
    beta, residuals = fm.rirls(X, y, M=st.bisquare, tune=4.685,
               scale_est=st.mad, scale_constant=0.6745, update_scale=True,
               maxiter=50, tol=1e-8)

    np.testing.assert_allclose(beta, np.array([[intercept, intercept],
                                               [slope, slope]]))


def test_roc_stable_fit(X_y_dates_romania):
    """Only integration test"""
    X, y, dates = X_y_dates_romania
    dates = dates.astype('datetime64[D]').astype('int')
    beta, resid, is_stable = fm.roc_stable_fit(X.astype(np.float32),
                                               y.astype(np.float32),
                                               dates)


