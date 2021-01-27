import numpy as np
import xarray as xr

import nrt.outliers as so


def test_screen_outliers_ccdc(X_y_clear):
    X, y, clear = X_y_clear

    green = xr.DataArray(y[:,:,np.newaxis])
    swir = xr.DataArray(y[:,:,np.newaxis])

    X = X.astype(np.float32)

    is_clear = so.ccdc_rirls(X=X, y=y, green=green, swir=swir)
    np.testing.assert_array_equal(~clear, np.isnan(is_clear))
