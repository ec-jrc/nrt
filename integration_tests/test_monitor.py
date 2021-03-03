from pathlib import Path
import pytest
import numpy as np

from nrt.monitor import iqr, ewma, cusum, mosum

monitor_params = {
    'EWMA': pytest.param(ewma.EWMA, {'trend': False, 'L': 5}, 5,
                         marks=pytest.mark.ewma),
    'IQR': pytest.param(iqr.Iqr, {'trend': False, 'harmonic_order': 1}, 3,
                        marks=pytest.mark.iqr),
    'CUSUM': pytest.param(cusum.CuSum, {}, 6,
                          marks=pytest.mark.cusum),
    'MOSUM': pytest.param(mosum.MoSum, {}, 6,
                          marks=pytest.mark.mosum)
}

@pytest.mark.parametrize('monitor_cls, kwargs, beta', monitor_params.values(),
                         ids=monitor_params.keys())
def test_fit_monitor(monitor_cls, kwargs, beta,
                     ndvi_history, ndvi_monitoring_numpy, forest_mask):
    monitor_ = monitor_cls(**kwargs, mask=forest_mask)
    monitor_.fit(dataarray=ndvi_history, **kwargs)
    assert monitor_.beta.shape[0] == beta
    for array, date in zip(*ndvi_monitoring_numpy):
        monitor_.monitor(array=array, date=date)
    monitor_._report()


@pytest.mark.parametrize('monitor_cls, kwargs, beta', monitor_params.values(),
                         ids=monitor_params.keys())
def test_netcdf(monitor_cls, kwargs, beta, ndvi_history, tmp_path):
    nc_path = tmp_path / 'monitor.nc'
    monitor_ = monitor_cls(**kwargs)
    monitor_.fit(dataarray=ndvi_history, **kwargs)

    monitor_.to_netcdf(nc_path)
    monitor_load = monitor_cls().from_netcdf(nc_path)
    assert monitor_ == monitor_load

