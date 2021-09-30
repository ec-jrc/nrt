import numpy as np

import nrt.outliers as so


def test_screen_outliers_ccdc(X_y_clear):
    X, y, clear = X_y_clear

    X = X.astype(np.float64)

    is_clear = so.ccdc_rirls(X=X, y=y, green=y, swir=y)
    np.testing.assert_array_equal(~clear, np.isnan(is_clear))
