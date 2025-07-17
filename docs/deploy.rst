.. _deploying_nrt:

Deploying your own operational NRT system
******************************************

Introduction
============

Monitoring algorithms implemented in the ``nrt`` python package are suitable for operational deployment. Setting up such a monitoring system requires at least some level of understanding of how the algorithms work as well as some IT skills; it would typically require the collaboration of a domain expert and an IT specialist. This page summarizes the main steps of the deployment process and provides useful snippets to facilitate it.

It assumes you have access to a constantly updated archive of Sentinel-2 L2A data indexed in a STAC catalogue and queryable via a STAC API.

The steps, which may differ depending on the specific objectives of your team, are:

* Define the study area, CRS, and tiling strategy.
* Obtain a forest mask.
* Define the monitoring strategy and optimal parameters.
* Fit the model(s) on a stable history period.
* **Fast forward** (optional): Bring the monitoring state from the end of the historical period to the present day.
* Set up periodic, automatic monitoring.
* Publish the output.
* Develop additional downstream services.

Study area, CRS, and tiling strategy
====================================

The size of the area you intend to monitor and your output format requirements will impact your choice of a Coordinate Reference System (**CRS**) and **tiling strategy**.

While a small area (e.g., less than 50x50 km) can be processed in a single chunk, it's highly recommended to split larger areas into multiple tiles. This allows for parallel processing and easier management. You may choose to handle the entire process in the native CRS of the Sentinel data you're using. However, if your study area crosses several UTM zones, it's often easier to define a common CRS for all processing and warp all input data to that single grid.

Obtaining a forest mask
=======================

It's highly recommended to use a **forest mask** to restrict monitoring to forested pixels. This avoids spurious anomaly detections in areas like agricultural fields or water bodies, which would add noise to the final product.

If a reliable tree cover layer is available for your area and matches your data's resolution, you can use that. Alternatively, you can quickly generate a forest mask using the ``xinfereo`` Python package.

Here is an example of producing a forest mask using ``xinfereo``:

.. code-block:: python

    import datetime

    import xarray as xr
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from affine import Affine
    from pystac_client import Client
    from odc.geo.geobox import GeoBox
    from odc.stac import stac_load

    from xinfereo import EOInferencer, data

    # 1. Define the Study area via a GeoBox
    gbox = GeoBox(
        (1000, 1000),
        Affine(20.0, 0.0, 3810180, 0.0, -20.0, 2517580),
        CRS.from_epsg(3035)
    )

    # 2. Query and load one year of Sentinel-2 data
    date_range = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 12, 31)]
    catalog = Client.open('https://your-stac-api.com/api')
    query = catalog.search(
        collections=["EO.Copernicus.S2.L2A"], # Or your collection name
        bbox=gbox.geographic_extent.boundingbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 50}}
    )

    # 3. Load the data into an xarray.Dataset
    ds = stac_load(
        query.items(),
        bands=['B02_20', 'B03_20', 'B04_20', 'B05', 'B06',
               'B07', 'B8A', 'B11', 'B12'],
        resampling='cubic',
        geobox=gbox,
        groupby='solar_day',
        chunks={'time': 1}
    )

    # 4. Apply standard offset and scaling
    ds = (ds.astype(np.float32) - 1000) / 10000

    # 5. Initialize the EOInferencer with a pre-trained model
    infrs = EOInferencer(data.tcd_model_meta(), 'tcd_onnx_v0.2')

    # 6. Run inference to predict tree cover density (returns a numpy array)
    tcd_predictions = infrs.predict(ds)

    # 7. Apply a threshold to create a binary forest/non-forest mask
    forest_mask = (tcd_predictions[0] > 20).astype(np.uint8)

    # 8. Save mask to disk for later use using metadata from the input xarray
    transform = ds.rio.transform()
    crs = ds.rio.crs
    with rasterio.open(
        "forest_mask.tif",
        'w',
        driver='GTiff',
        height=forest_mask.shape[0],
        width=forest_mask.shape[1],
        count=1,
        dtype=forest_mask.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(forest_mask, 1)


Define monitoring strategy and optimal parameters
=================================================

This step requires experimentation to find the best configuration for your specific use case. A non-exhaustive list of parameters to choose from is:

* **Vegetation Index** (e.g., NDVI, NBR, NDMI)
* **Monitoring algorithm** (CCDC, EWMA, CuSum, MoSum, and IQR are currently implemented in ``nrt``)
* **Fitting parameters** (e.g., fitting algorithm, outlier removal strategy)
* **Monitoring parameters** (e.g., warming-up period, alert thresholds)

There are two main approaches to select the optimal parameters:

1.  **Using reference data**: This is the most objective method. If you have reference data from field surveys or visual interpretation, you can use them to quantitatively assess the performance of different parameter combinations. The ``nrt-validate`` package provides tools like the ``SegmentInterpreter`` to facilitate this process. `See nrt-validate documentation <https://nrt-validate.readthedocs.io/en/latest/api_reference.html>`_.
2.  **Simulation on a test area**: A more qualitative approach is to run simulations on a well-known test area of manageable size. While less objective, this will give you a good intuitive feel for how well the results match the known disturbance patterns of the area.

Fit the stable history period for the chosen model(s)
=====================================================

Once you've chosen a model and its parameters, the first step is to fit it on a "stable" historical period. This period should contain representative, cloud-free observations of the "normal" forest phenology, without major disturbances.

The following snippet shows how to fit an ``IQR`` monitor with an ``NDVI`` time series from 2022 to 2024.

.. code-block:: python

    import datetime

    import xarray as xr
    import numpy as np
    from rasterio.crs import CRS
    from affine import Affine
    from pystac_client import Client
    from odc.geo.geobox import GeoBox
    from odc.stac import stac_load
    from nrt.monitor.iqr import IQR

    # 1. Define the same study area and historical period
    gbox = GeoBox(
        (1000, 1000),
        Affine(20.0, 0.0, 3810180, 0.0, -20.0, 2517580),
        CRS.from_epsg(3035)
    )
    history_period = [datetime.datetime(2022, 1, 1), datetime.datetime(2024, 12, 31)]

    # 2. Query STAC for the historical data, including the SCL band for masking
    catalog = Client.open('https://your-stac-api.com/api')
    query = catalog.search(
        collections=["EO.Copernicus.S2.L2A"],
        bbox=gbox.geographic_extent.boundingbox,
        datetime=history_period,
        query={"eo:cloud_cover": {"lt": 80}} # Higher cloud cover is ok, we will mask it
    )

    # 3. Load data
    ds = stac_load(
        query.items(),
        bands=['B04_20', 'B8A', 'SCL'],
        resampling={'B04_20': 'cubic',
                    'B8A': 'cubic',
                    'SCL': 'nearest'},
        geobox=gbox,
        groupby='solar_day',
        chunk={'time': 1}
    )

    # 4. Pre-process the data
    ds = ds.where(ds != 0)
    ds['B04_20'] = ds.B04_20 - 1000
    ds['B8A'] = ds.B8A - 1000
    # Mask clouds and non-vegetation pixels
    ds = ds.where(ds.SCL.isin([4, 5, 6, 7]))
    ds = ds.drop_vars('SCL')

    # 5. Compute NDVI
    ndvi = (ds.B8A - ds.B04_20) / (ds.B8A + ds.B04_20)
    ndvi = ndvi.where(~np.isinf(ndvi), np.nan) # The above division may generate inf values which unlike nan are not supported by nrt's fit function 
    ndvi = ndvi.compute()

    # 6. Initialize IQR monitor
    with rasterio.open('forest_mask.tif') as src:
        mask = src.read(1)
    model = IQR(mask=mask, trend=False, harmonic_order=2, sensitivity=2.5)

    # 7. Fit the model on the historical NDVI data
    model.fit(ndvi, method='RIRLS', maxiter=5)

    # 8. Save the fitted state to disk for the next steps
    model.to_netcdf("./data/iqr_ndvi_fitted_state.nc")

    print("Fitting complete. State saved to disk.")


Fast forward
============

If the end of your historical period does not match today's date, you may want to "fast forward" the monitoring state. This involves iteratively applying the ``monitor`` method for all observations between the end of the fit period and the present. This is also a good strategy for algorithms that require a "warm-up" period to perform optimally.

The following snippet loads the state from the previous step and updates it with data up to July 15, 2025.

.. code-block:: python

    import os
    import datetime

    import numpy as np
    from pystac_client import Client
    from odc.stac import stac_load
    from nrt.monitor.iqr import IQR
    # Assume gbox is defined or loaded from a configuration file

    # 1. Configuration
    STATE_FILE_INPUT = "./data/iqr_ndvi_fitted_state.nc"
    STATE_FILE_OUTPUT = "./data/iqr_ndvi_current_state.nc"
    REPORT_FILE = "./data/nrt_report.tif"
    DATE_LOG_FILE = "./data/processed_dates.log"
    FF_PERIOD = [datetime.datetime(2025, 1, 1), datetime.datetime(2025, 7, 16)]

    # Note: Before running, you may want to manually remove the old DATE_LOG_FILE
    # to ensure the fast-forward process starts with a clean slate.

    # 2. Load the previously fitted monitor state
    model = IQR.from_netcdf(STATE_FILE_INPUT)
    print("Loaded fitted state from:", STATE_FILE_INPUT)

    # 3. Query STAC for the fast-forward period
    catalog = Client.open('https://your-stac-api.com/api')
    query = catalog.search(
        collections=["EO.Copernicus.S2.L2A"],
        bbox=gbox.geographic_extent.boundingbox,
        datetime=FF_PERIOD
    )
    items = list(query.items())

    if items:
        print(f"Found {len(items)} items for the fast-forward period.")

        # 4. Load data and calculate NDVI
        ds = stac_load(items,
                       bands=['B04_20', 'B8A', 'SCL'],
                       resampling={'B04_20': 'cubic',
                                   'B8A': 'cubic',
                                   'SCL': 'nearest'},
                       geobox=gbox,
                       groupby='solar_day',
                       chunks={'time': 1})
        ds = ds.where(ds != 0)
        ds['B04_20'] = ds.B04_20 - 1000
        ds['B8A'] = ds.B8A - 1000
        ds = ds.where(ds.SCL.isin([4, 5, 6, 7]))
        ds = ds.drop_vars('SCL')

        # 5. Compute NDVI
        ndvi = (ds.B8A - ds.B04_20) / (ds.B8A + ds.B04_20)
        ndvi = ndvi.where(~np.isinf(ndvi), np.nan)
        ndvi = ndvi.sortby('time')
        ndvi = ndvi.compute()

        # 5. Iteratively monitor each new observation and log the date
        for date in ndvi.time.values.astype('M8[s]').astype(datetime.datetime):
            ndvi_slice = ndvi.sel(time=date, method='nearest').values
            if not np.isnan(ndvi_slice).all():
                model.monitor(array=ndvi_slice, date=date)

                # For logging, convert the date to a simple string
                date_str = date.strftime('%Y-%m-%d')
                with open(DATE_LOG_FILE, 'a') as log:
                    log.write(date_str + "\n")
                print(f"Processed and logged date: {date_str}")

        # 6. Generate the final report
        model.report(
            filename=REPORT_FILE,
            layers=['mask', 'detection_date'],
            dtype=np.uint8
        )
        print("Report generated at:", REPORT_FILE)

        # 7. Save the final, updated state to disk
        model.to_netcdf(STATE_FILE_OUTPUT)
        print("Fast forward complete. Final state saved to:", STATE_FILE_OUTPUT)

    else:
        print("No new data found for the fast forward period.")


Automatic monitoring
====================

After fitting and fast-forwarding, the system is ready for operational, automatic execution. This involves running a script periodically (e.g., daily) to check for new satellite imagery and update the monitoring state.

Monitoring Snippet
------------------

This generic script loads the current state, checks for new data since the last run, processes it, and saves the updated state and reports.

.. code-block:: python

    # operational_monitoring.py
    import datetime

    import os
    import numpy as np
    from pystac_client import Client
    from odc.stac import stac_load
    from nrt.monitor.iqr import IQR
    # Assume gbox is defined or loaded from a config

    def get_last_date_from_log(log_file, initial_date):
        """Reads the last date from the log file, returns initial_date if not found."""
        if not os.path.exists(log_file):
            return initial_date
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                return initial_date
            return datetime.datetime.fromisoformat(lines[-1].strip())

    def run_daily_monitoring():
        STATE_FILE = "./data/iqr_ndvi_current_state.nc"
        REPORT_FILE = "./data/nrt_report.tif"
        DATE_LOG_FILE = "./data/processed_dates.log"

        # The initial date should be the last day of your fitting/fast-forward period.
        # This is only used if the log file doesn't exist.
        initial_start_date = datetime.datetime(2025, 7, 15)

        # 1. Load the current monitoring state
        model = IQR.from_netcdf(STATE_FILE)

        # 2. Determine the date of the last observation from the log file
        last_date = get_last_date_from_log(DATE_LOG_FILE, initial_start_date)
        start_date = last_date + datetime.timedelta(days=1)
        end_date = datetime.datetime.now()

        print(f"Checking for new data from {start_date.date()} to {end_date.date()}")

        # 3. Query STAC for new data
        catalog = Client.open('https://your-stac-api.com/api')
        query = catalog.search(
            collections=["EO.Copernicus.S2.L2A"],
            bbox=gbox.geographic_extent.boundingbox,
            datetime=[start_date, end_date]
        )
        items = list(query.items())

        # 4. If new data is found, process it
        if not items:
            print("No new items since last update.")
            return

        print(f"Found {len(items)} new items. Processing...")
        # Same preprocessing as before
        ds = stac_load(items, bands=['B04_20', 'B8A', 'SCL'],
                       geobox=gbox,
                       groupby='solar_day',
                       chunks={'time': 1})
        ds = ds.where(ds != 0)
        ds['B04_20'] = ds.B04_20 - 1000
        ds['B8A'] = ds.B8A - 1000
        ds = ds.where(ds.SCL.isin([4, 5, 6, 7]))
        ds = ds.drop_vars('SCL')

        # 5. Compute NDVI
        ndvi = (ds.B8A - ds.B04_20) / (ds.B8A + ds.B04_20)
        ndvi = ndvi.where(~np.isinf(ndvi), np.nan)
        ndvi = ndvi.sortby('time')
        ndvi = ndvi.compute()

        # 5. Monitor, log date, report, and save
        new_dates_processed = False
        for date in ndvi.time.values.astype('M8[s]').astype(datetime.datetime):
            # Convert numpy.datetime64 to python datetime for logging
            date_str = date.strftime('%Y-%m-%d')
            ndvi_slice = ndvi.sel(time=date, method='nearest').values
            if not np.isnan(ndvi_slice).all():
                model.monitor(array=ndvi_slice, date=date)
                # Log the date immediately after it has been processed
                with open(DATE_LOG_FILE, 'a') as log:
                    log.write(date_str + "\n")
                new_dates_processed = True

        # 6. Only update reports and state if new data was actually processed
        if new_dates_processed:
            model.report(filename=REPORT_FILE, layers=['mask', 'detection_date'], dtype=np.uint8)
            model.to_netcdf(STATE_FILE) # Overwrite the state file with the updated version
            print("Monitoring update complete.")
        else:
            print("No new valid observations found to process.")

    if __name__ == "__main__":
        run_daily_monitoring()

Automatic execution setup
-------------------------

A common approach for automating this script is using **CRON** to schedule its execution within a **Docker** container. This ensures a consistent and isolated environment. The specific method for scheduling and orchestration, however, is typically a decision for the IT specialist or system administrator managing the deployment.

1.  **Dockerfile**: Create a ``Dockerfile`` to build an image with all necessary dependencies.

.. code-block:: dockerfile

    FROM python:3.10-slim
    WORKDIR /app
    COPY . .
    RUN pip install --no-cache-dir nrt odc-stac odc-geo pystac-client rioxarray xarray
    CMD ["python", "operational_monitoring.py"]

2.  **CRON Job**: Add a line to your ``crontab`` to run the Docker container daily.

.. code-block:: bash

    # Edit crontab with: crontab -e

    # Run the monitoring script every day at 2:00 AM
    0 2 * * * docker run --rm -v /path/to/your/data:/app/data your-nrt-image-name

This command tells CRON to run your Docker container at 2 AM. The ``-v`` flag mounts your local data directory into the container, allowing the script to read the state file and write the outputs.

Publish the output
==================

Once you have your report products (e.g., disturbance mask, detection date), you'll want to make them accessible.

* The **Web Map Service (WMS)** protocol is a good option for serving the outputs as map layers that can be consumed by GIS software (QGIS, ArcGIS) or web clients.
* If you adopted a tiling strategy, creating a **VRT (Virtual Raster)** is a convenient way to mosaic all tiles into a single virtual layer. This allows individual tiles to be updated without needing to rewrite the entire mosaic, making the update process much faster.

Considerations
==============

.. note::
    Logging: It is crucial to implement comprehensive logging in your operational scripts. This will help you debug issues when the script runs unattended.

.. note::
    State Backups: The operational script overwrites the state file on each successful run. It's wise to implement a backup strategy (e.g., renaming the old state file with a timestamp before saving the new one) to prevent a failed run from corrupting your only state file.

.. note::
    Filesystem Stability: Depending on the architecture you are deploying on (e.g., HPC, cloud), writing directly to a distributed or network file system can be risky. An interrupted process could leave files in a corrupted or incomplete state. A more stable pattern is to write all outputs to a temporary local disk first (e.g., ``/scratch``). Once all processing is complete, a single, atomic move operation can transfer the final products to the persistent storage.

.. note::
    Parallel Processing with Dask: For large-scale deployments, the data loading strategy presented above may be inneficient. Dask provides a powerful solution for parallel and out-of-core (lazy) computation. When using ``odc.stac.stac_load``, you can pass a chunks argument (e.g., chunks={'x': 1024, 'y': 1024}). This instructs stac_load to return a Dask array instead of a NumPy array, allowing data to be loaded, processed, and written in parallel blocks.