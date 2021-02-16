from nrt.monitor.ccdc import CCDC

# For now, because fixtures can't be parametrized and CCDC needs
# green_swir_history, this is done seperately. The package pytest-lazy-fixture
# could be used to also parametrize CCDC.


def test_fit_monitor(ndvi_history, green_swir_history, ndvi_monitoring_numpy):
    green, swir = green_swir_history
    ccdc_monitor = CCDC()
    ccdc_monitor.fit(dataarray=ndvi_history, green=green, swir=swir,
                     scaling_factor=10000)
    assert ccdc_monitor.beta.shape[0] == 6 # 2*2 harmonics + intercept + trend
    for array, date in zip(*ndvi_monitoring_numpy):
        ccdc_monitor.monitor(array=array, date=date)
    ccdc_monitor._report()


def test_netcdf(ndvi_history, green_swir_history, tmp_path):
    nc_path = tmp_path / 'ccdc.nc'
    green, swir = green_swir_history
    ccdc_monitor = CCDC()
    ccdc_monitor.fit(dataarray=ndvi_history, green=green, swir=swir,
                     scaling_factor=10000)

    ccdc_monitor.to_netcdf(nc_path)
    CCDC.from_netcdf(nc_path)
