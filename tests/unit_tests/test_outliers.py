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

import nrt.outliers as so


def test_screen_outliers_ccdc(X_y_clear):
    X, y, clear = X_y_clear

    X = X.astype(np.float64)

    is_clear = so.ccdc_rirls(X=X, y=y, green=y, swir=y)
    np.testing.assert_array_equal(~clear, np.isnan(is_clear))
