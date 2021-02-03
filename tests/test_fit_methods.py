import numpy as np

import nrt.fit_methods as fm
import nrt.stats
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


def test_ccdc_is_stable(stability_ccdc, threshold=3):
    residuals, slope, check_stability = stability_ccdc
    is_stable = nrt.stats.is_stable_ccdc(slope, residuals, threshold)
    np.testing.assert_array_equal(is_stable, check_stability)


def test_recresid(X_y_dates_romania, strcchng_rr):
    X, y, dates = X_y_dates_romania
    result = strcchng_rr
    span = X.shape[1]+1

    rresid = st.recresid(X=X, y=y, span=span)

    # remove span from results and compare. Slight differences in the beginning
    # because of floating point accuracy
    np.testing.assert_allclose(rresid[span+2:,:], result[span+2:,:])


def test_roc_stable_fit(X_y_dates_romania):
    X, y, dates = X_y_dates_romania
    beta, resid, is_stable = fm.roc_stable_fit(X, y)

