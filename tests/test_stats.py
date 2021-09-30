from scipy.stats import norm
import numpy as np
import pytest

import nrt.stats as st


# validate ncdf against scipy norm.cdf
@pytest.mark.parametrize("test_input", [-1,1,0.3])
def test_ncdf(test_input):
    scipy_result = norm.cdf(test_input)
    numba_result = st.ncdf(test_input)
    np.testing.assert_allclose(numba_result, scipy_result)


def test_ccdc_is_stable(stability_ccdc, threshold=3):
    residuals, slope, check_stability = stability_ccdc
    is_stable = st.is_stable_ccdc(slope, residuals, threshold)
    np.testing.assert_array_equal(is_stable, check_stability)


def test_nan_percentile_axis0():
    # test data
    xy = (20, 20, 20)
    test_data = np.random.random_sample(xy)
    # turn 10% into nan
    rand_nan = np.random.random_sample(xy) < 0.1
    test_data[rand_nan] = np.nan

    q_numba = st.nan_percentile_axis0(test_data, np.array([75, 25]))
    q_numpy = np.nanpercentile(test_data, [75 ,25], 0)

    np.testing.assert_allclose(q_numba, q_numpy)
