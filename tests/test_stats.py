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