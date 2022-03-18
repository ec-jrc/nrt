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
