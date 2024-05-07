---
title: 'NRT: operational monitoring of satellite images time-series in python'
tags:
  - Python
  - time-series
  - monitoring
  - spatial
  - forest
  - geosciences
authors:
  - name: Loïc Dutrieux
    orcid: 0000-0002-5058-2526
    equal-contrib: true
    affiliation: 1
  - name: Jonas Viehweger
    orcid: 0000-0002-1610-4600
    equal-contrib: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
affiliations:
 - name: European Commission, Joint Research Centre, Ispra, Italy
   index: 1
 - name: Philipp University of Marburg, RWTH Aachen University, Germany
   index: 2
 - name: Sinergise Solutions GmbH, Graz, Austria
   index: 3
date: 7 May 2024
bibliography: refs.bib
---


<!--
  Not part of the paper
  - Explicit reference to offline time-series approaches (breakpoints, LandTrendr)
  - Use of nrt in Jonas' MSc thesis for intercomparison of monitoring approaches
  - 
-->

# Abstract

`nrt` is a Python package specifically designed for near real-time detection of anomalies in gridded spatio-temporal data, and particularly suited for monitoring forest disturbances from satellite image time-series.
The core of the package provides a standardized Application Programming Interface (API) that facilitates interoperability and comparison across a variety of monitoring algorithms.
These algorithms, selected from the scientific literature, are implemented in an optimized manner to enable efficient computation across large geographical areas.
This development represents a significant advancement in environmental monitoring efforts, enhancing our ability to contribute to climate change mitigation, biodiversity conservation, and the preservation of natural heritage.



# Statement of need

Accurate mapping and monitoring of land dynamics is critical for climate change mitigation, biodiversity conservation, and epidemic prevention [@foresteurope2020state].
With increasing data availability and processing capacities, there is a growing desire to move from periodic mapping of forest disturbances to continuous monitoring systems capable of providing timely information on forest disturbances to a variety of stakeholders [@woodcock2020transitioning].
Many algorithms and approaches have been proposed in the research community to address this near real-time monitoring challenge.
Their performance is typically demonstrated based on case studies over test areas or simulated datasets.
However, when it is available, the software provided with the research papers only offers limited operational capacity.
Individual software is often primarily developed to support the research experiment and consequently not optimized for speed or deployment at scale.
In addition, implementation in different programming languages or the absence of a common interface to operate the algorithm make interoperability and comparisons exercises challenging.
The `nrt` package directly addresses these shortcomings by providing a standardized Application Programming Interface (API) to various near real-time disturbance detection approaches, all optimized for rapid computation and specifically designed for effective operational deployment.



# Online monitoring of time-series data

The term 'online monitoring' refers to the real-time analysis of open-ended time-series that grow dynamically, with each new data point updating the analysis values in real-time.
This method contrasts with 'offline' approaches, which involve retrospective analysis of static time-series data.
Online monitoring is particularly suited for applications requiring immediate responses, such as anomaly detection in environmental monitoring or system health tracking.
In the context of satellite imagery, where spaceborne sensors such as Sentinel 2 and Landsat -- the main optical moderate resolution constellations -- collect data with a revisit period of 5 and 8 days respectively near the equator, each pixel is treated as an independent online time-series.
Various vegetation indices, indicative of traits such as photosynthetic activity, foliage density, or water content, are derived from raw reflectance data to monitor stability as a univariate time-series.
To establish stable behavior, statistical models are employed; often, linear harmonic regressions are used to model typical seasonal behavior over what is termed the stable history period.
Disturbances in vegetation, such as removal due to logging or stress from biotic and abiotic factors like insects, diseases, and droughts, typically manifest as anomalies in these time-series (\autoref{fig:concept}).

![Illustration of the online monitoring process for a univariate time-series, depicting the distinct phases including the history period, monitoring period, and a detected break against modeled values. \label{fig:concept}](figs/NRT_image.png)

Different strategies for detecting these anomalies define the unique monitoring approaches implemented in the `nrt` package.
The operational deployment of online monitoring systems, particularly for satellite imagery, requires efficient data management strategies.
Since keeping monitoring processes in memory would tie up computing resources for prolonged periods, these processes must instead be saved on disk.
They are then reloaded and updated periodically as new satellite data becomes available.


# Main functionalities

## A common monitoring framework

The monitoring algorithms in `nrt` are structured as subclasses of an abstract base class that defines a general monitoring framework.
This framework is designed to operate with xarray DataArray data structures and includes several key methods:

  - `fit`: This method is used for model fitting over the stable history period, employing temporal regression with harmonic components. This step is typically the most computationally intensive, but efficiency is enhanced through vectorized operations and the use of the numba Just In Time (JIT) compiler [@lam2015numba]. Most fitting processes can also be parallelized directly via the API.
  - `monitor`: Updates the monitoring status with the latest available data.
  - `report`: Exports variables of interest from the monitoring status into a georeferenced raster file.
  - `to/from_netcdf`: Manages the dumping and loading of the current monitoring status to and from disk for persistent storage and agile monitoring.


The framework also defines several noteworthy attributes:

  - `mask`: A two-dimensional spatial array that provides high-level information about the monitoring status, with different integer labels denoting whether pixels are currently being monitored, unstable, no longer monitored, etc.
  - `beta`: Represents the regression coefficient matrix from the temporal linear regression.
  - `process` and `boundary`: These attributes slightly vary depending on the monitoring approach but generally relate to the dynamic values updated during monitoring. Typically, a process value crossing the boundary indicates the detection of a breakpoint, triggering further actions in the monitoring workflow.


## State of the art monitoring algorithm implemented

The `nrt` package currently implements five distinct monitoring approaches:

- Exponentially Weighted Moving Average (EWMA) [@brooks2013fly].
- Cumulative Sum of Residual (CuSum) [@verbesselt2012near; @zeileis2005monitoring]. CuSum is a monitoring option in the bfastmonitor function of the R package bfast.
- Moving Sum of Residuals (MoSum) [@verbesselt2012near; @zeileis2005monitoring]. Like CuSum, MoSum is available through the bfastmonitor function in bfast.
- Continuous Change Detection and Classification of land cover (CCDC, CMFDA) [@zhu2012continuous; @zhu2014continuous] - Focuses on the core change detection component of the original CCDC algorithm.
- InterQuantile Range (IQR) - A simple, unpublished outlier identification strategy proposed by Rob J. Hyndman on [StackExchange](https://stats.stackexchange.com/a/1153).

While all temporal fitting methods are compatible with each monitoring algorithm, the default settings are tailored independently for each approach, adhering to the original manuscripts' recommendations. For example, @verbesselt2012near recommend performing a stability test on the historical period to ensure no disturbances are present; thus, the default fitting strategy for CuSum and MoSum is `ROC` (Reversed Cumulative Sum of residuals). However, if the historical period is known to be stable and free of disturbances, users may opt for a simpler and faster fitting strategy such as `OLS` (Ordinary Least Squares).


## Demo data

The `nrt` package features a data module that provides access to both real and simulated datasets.
Included in the module is a subset of a Sentinel 2 multi-temporal data cube, completed with an associated forest mask.
This real data set is ideal for demonstration purposes.

Additionally, the data simulation functionalities of the module allow for the creation of synthetic time-series.
Users can fully control various parameters, such as seasonal amplitude, noise level, frequency of outliers, and the magnitude of breaks.
These capabilities are especially valuable for development, testing, and conducting sensitivity analyses.


# Current use

The most significant application of the `nrt` package to date is its deployment as a prototype alert system at the Joint Research Center.
This system ran operationally throughout the 2023 vegetation season, covering the entire territory of Estonia (\autoref{fig:deployment}).
The system utilizes the EWMA algorithm from `nrt`, minimally tuned, and employs Normalized Difference Vegetation Index (NDVI) and Normalized Difference Moisture Index (NDMI) indices to generate forest disturbance alerts.

![Snapshot of the operational forest disturbance alert system deployed across Estonia during the 2023 vegetation period. The bottom-right panel shows an example of a tree cover loss event that was successfully detected. \label{fig:deployment}](figs/deployment.png)

Estonia was divided into 50 km tiles, and a monitoring instance for each tile was established using data from 2021 and 2022 as a stable history period.
Each night, a containerized process automatically queried the SpatioTemporal Asset Catalog (STAC) on the Big Data Analytics Platform (BDAP) platform for new Sentinel 2 data [@soille2018versatile; @STAC2021]. When new data was available, it was used to update both the monitoring status and the alert layer, which was accessible via Web Map Service (WMS) protocol.

It is important to note that tools like `PySTAC` and Open Data Cube (ODC) played a crucial role in the deployment and functionality of this pipeline [@hanson2019open; @killough2018overview].

<!--
The package also served the work of a MSc thesis on intercomparison of multiple forest monitoring approaches, illustrating the dual purpose of the package to ease intercomparison while being fully deployable at scale.
-->


# Future directions

Active development on `nrt` and its ecosystem continues, and we warmly welcome external contributors. The canonical repository for the core `nrt` package is hosted on the [ec-jrc organization on GitHub](https://github.com/ec-jrc) and will remain there, while future additions such as namespace packages or additional material will be hosted on [code.europa.eu/jrc-forest](code.europa.eu/jrc-forest).
Our strategic vision aims to expand both the core functionalities of the package and its surrounding ecosystem, aspiring to establish a unified approach to forest disturbance monitoring. Currently, a multivariate monitoring mode is being designed to enhance the core functionalities. This new mode aims to mitigate the trade-offs typically faced when selecting a vegetation index for disturbance detection, where indices sensitive to greenness for instance are generally less sensitive to water content and vice versa.
Developments in progress and envisioned for the `nrt` ecosystem include tools and tutorials for visualization and diagnostics, computation of accuracy metrics, the creation of reference data through visual interpretation, and advanced generation of synthetic data.

# Conclusion

The `nrt` package marks a significant advancement in near real-time monitoring of forest disturbances. By optimizing state-of-the-art algorithms and providing a standardized API, it greatly enhances interoperability and enables scalable deployment. These enhancements substantially benefit environmental monitoring efforts, contributing to climate change mitigation, biodiversity conservation, and the preservation of natural heritage. Future developments will focus on expanding its capabilities and usability, aiming to develop a unified approach to monitoring forest disturbances.


# References