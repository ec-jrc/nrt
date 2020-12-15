import datetime
from math import pi

import pandas as pd
import numpy as np


def build_regressors(dates, trend=True, harmonic_order=3):
    """Build the design matrix (X) from a list or an array of datetimes

    Trend assumes temporal resolution no finer than daily
    Harmonics assume annual cycles

    Args:
        dates (pandas.DatetimeIndex): The dates to use for building regressors
        trend (bool): Whether to add a trend component
        harmonic_order (int): The order of the harmonic component

    Returns:
        numpy.ndarray: A design matrix
    """
    dates = dates.sort_values()
    shape = (len(dates), 1 + trend + 2*harmonic_order)
    X = np.zeros(shape, dtype=np.float)
    # Add intercept (Is that actually required?)
    X[:,0] = 1
    if trend:
        origin = pd.Timestamp(1970)
        X[:,1] = (dates - origin).days
    if harmonic_order:
        indices = range(1 + trend, 1 + trend + 2 * harmonic_order)
        # Array of decimal dates
        ddates = datetimeIndex_to_decimal_dates(dates)
        # Allocate array
        X_harmon = np.empty((len(dates), harmonic_order))
        for i in range(harmonic_order):
            X_harmon[:,i] = 2 * np.pi * ddates * (i + 1)
        X_harmon = np.concatenate([np.cos(X_harmon), np.sin(X_harmon)], 1)
        X[:, indices] = X_harmon
    return X


def dt_to_decimal(dt):
    """Helper to build a decimal date from a datetime object
    """
    year = dt.year
    begin = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year, 12, 31)
    return year + (dt - begin)/(end - begin)


def datetimeIndex_to_decimal_dates(dates):
    """Convert a pandas datetime index to decimal dates"""
    years = dates.year
    first_year_day = pd.to_datetime({'year':years, 'day':1, 'month':1})
    last_year_day = pd.to_datetime({'year':years, 'day':31, 'month':12})
    ddates = years + (dates - first_year_day)/(last_year_day - first_year_day)
    return np.array(ddates, dtype=np.float)
