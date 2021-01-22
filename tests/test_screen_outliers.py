import nrt.screen_outliers as so
import numpy as np

def test_screen_outliers_ccdc(X_y_clear):
    X, y, clear = X_y_clear

    is_clear = so.ccdc_rirls(X=X, y=y, green=y, swir=y)
    np.testing.assert_array_equal(~clear, np.isnan(is_clear))