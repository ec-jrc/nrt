import numpy as np
import pytest

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
    #np.testing.assert_allclose(np.dot(X, beta), y)



# @pytest.mark.parametrize(('X', 'y'), [
#     (np.random.rand(n, n), np.random.rand(n))
#     for n in range(1, 10)
# ])
# def test_RLM_issue88(X, y):
#     """ Issue 88: Numeric problems when n_obs == n_reg/k/p/number of regressors
#     The regression result will be garbage so we're not worrying about the
#     coefficients. However, it shouldn't raise an exception.
#     """
#     beta, residuals = fm.rirls(X, y, M=st.bisquare, tune=4.685,
#                scale_est=st.mad, scale_constant=0.6745, update_scale=True,
#                maxiter=50, tol=1e-8)


def test_ccdc_is_stable(stability_ccdc, threshold=3):
    residuals, slope, check_stability = stability_ccdc

    is_stable = fm.ccdc_is_stable(slope,residuals,threshold)
    np.testing.assert_array_equal(is_stable, check_stability)


def test_screen_outliers_ccdc(X_y_clear):
    X, y, clear = X_y_clear

    is_clear = fm.screen_outliers_rirls(X, y, y)
    np.testing.assert_array_equal(clear, is_clear)


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
    y[0,9] = 0
    y[1,0] = 23
    # y[0,5] = np.nan
    # y[1,5] = np.nan

    return X, y.T, intercept, slope


# fixture of 2D residuals with extreme values start and end
# 1D slope with extreme value and corresponding results in stability
@pytest.fixture
def stability_ccdc(request):
    np.random.seed(0)
    # build an example, where one pixel has a large first value,
    # one a large last value one a large slope and one just random residuals
    residuals = np.random.rand(20,4)-0.5
    residuals[0,0] = 100
    residuals[-1,1] = 100
    slope = np.array([0,0,10,0])

    stability = np.array([False, False, False, True])

    return residuals, slope, stability