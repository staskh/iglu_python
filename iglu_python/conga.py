from typing import Union

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns


def conga(data: Union[pd.DataFrame, pd.Series], n: int = 24, tz: str = "") -> pd.DataFrame | float:
    """
    Calculate Continuous Overall Net Glycemic Action (CONGA).

    The function produces CONGA values for any n hours apart. CONGA is the standard
    deviation of the difference between glucose values that are exactly n hours apart.

    Missing values will be linearly interpolated when close enough to non-missing values.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    n : int, default=24
        Number of hours between glucose measurements to use in CONGA calculation
    tz : str, default=""
        Time zone to use for datetime conversion. Empty string means use local time zone.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - CONGA: CONGA value (standard deviation of differences between measurements n hours apart)

    References
    ----------
    McDonnell et al. (2005) A novel approach to continuous glucose analysis
    utilizing glycemic variation
    Diabetes Technology and Therapeutics 7:253-263,
    doi:10.1089/dia.2005.7.253.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> conga(data)
       id      CONGA
    0  subject1  35.355
    1  subject2  42.426

    >>> conga(data['gl'], n=12)
       CONGA
    0  35.355
    """

    # Handle Series input
    if isinstance(data, (pd.Series)):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        return conga_single(data, hours=n, tz=tz)

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate CONGA for each subject
    data.set_index("time", inplace=True, drop=True)
    out = data.groupby("id").agg(CONGA=("gl", lambda x: conga_single(x, hours=n, tz=tz))).reset_index()

    return out


def conga_single(data: pd.DataFrame | pd.Series, hours: int = 1, tz: str = "") -> float:
    """Calculate CONGA for a single subject"""
    # Convert data to day-by-day format
    # Missing values will be linearly interpolated when close enough to non-missing values.
    gl_by_id_ip, _, dt0 = CGMS2DayByDay(data, tz=tz)

    # Calculate number of readings per hour
    hourly_readings = round(60 / dt0)

    # Calculate differences between measurements n hours apart
    # Flatten the matrix and calculate differences with lag
    gl_vector = gl_by_id_ip.flatten()

    # Calculate differences between measurements n hours apart
    # Flatten the matrix and calculate differences with lag
    lag = hourly_readings * hours
    diffs = gl_vector[lag:] - gl_vector[:-lag]

    # Check if we have sufficient data for std calculation
    # Need at least 2 non-NaN values for ddof=1
    valid_diffs = diffs[~np.isnan(diffs)]
    if len(valid_diffs) < 2:
        return np.nan

    return float(np.nanstd(diffs, ddof=1))
