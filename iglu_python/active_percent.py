from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import check_data_columns, localize_naive_timestamp


def active_percent(
    data: Union[pd.DataFrame, pd.Series],
    dt0: Optional[int] = None,
    tz: str = "",
    range_type: str = "automatic",
    ndays: int = 14,
    consistent_end_date: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame|dict[str:float]:
    """
    Calculate percentage of time CGM was active.

    The function produces a DataFrame with values equal to the percentage of time
    the CGM was active, the total number of observed days, the start date, and the end date.
    For example, if a CGM's (5 min frequency) times were 0, 5, 10, 15 and glucose values
    were missing at time 5, then percentage of time the CGM was active is 75%.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    dt0 : Optional[int], default=None
        Time interval in minutes between measurements. If None, it will be automatically
        determined from the median time difference between measurements.
    tz : str, default=""
        Time zone to be used. Empty string means current time zone, "GMT" means UTC.
    range_type : str, default="automatic"
        Type of range calculation ('automatic' or 'manual').
    ndays : int, default=14
        Number of days to consider in the calculation.
    consistent_end_date : Optional[Union[str, datetime]], default=None
        End date to be used for every subject. If None, each subject will have their own end date.
        Used only in range_type=='manual' mode

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier
        - active_percent: percentage of time CGM was active (0-100)
        - ndays: number of days of measurements
        - start_date: start date of measurements
        - end_date: end date of measurements

    References
    ----------
    Danne et al. (2017) International Consensus on Use of
    Continuous Glucose Monitoring
    Diabetes Care 40:1631-1640,
    doi:10.2337/dc17-1600.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:10:00', '2020-01-01 00:00:00',
    ...                            '2020-01-01 00:05:00']),
    ...     'gl': [150, np.nan, 160, 140, 145]
    ... })
    >>> active_percent(data)
       id  active_percent  ndays           start_date             end_date
    0  subject1      66.67    0.0  2020-01-01 00:00:00  2020-01-01 00:10:00
    1  subject2     100.00    0.0  2020-01-01 00:00:00  2020-01-01 00:05:00
    """

    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        return active_percent_single(data, dt0, tz, range_type, ndays, consistent_end_date)

    # Check data format and convert time to datetime
    data = check_data_columns(data, tz)

    # Initialize result list
    active_perc_data = []

    # Process each subject
    for subject in data["id"].unique():
        # Filter data for current subject and remove NA values
        sub_data = (
            data[data["id"] == subject]
            .dropna(subset=["gl", "time"])
            .sort_values("time")
        )

        timeseries = sub_data.set_index("time")["gl"]
        active_percent_dict = active_percent_single(timeseries, dt0, tz, range_type, ndays, consistent_end_date)
        active_percent_dict["id"] = subject
        active_perc_data.append(active_percent_dict)


    # Convert to DataFrame
    df = pd.DataFrame(active_perc_data)
    df = df[['id'] + [col for col in df.columns if col != 'id']]
    return df


def active_percent_single(
    data: pd.Series,
    dt0: Optional[int] = None,
    tz: str = "",
    range_type: str = "automatic",
    ndays: int = 14,
    consistent_end_date: Optional[Union[str, datetime]] = None,
) -> dict[str:float]:
    """
    Calculate percentage of time CGM was active for a single series/subject.
    """

    if not isinstance(data, pd.Series):
        raise ValueError("Input must be a Series")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    data = data.dropna()
    if len(data) == 0:
        return {"active_percent": 0, "ndays": 0, "start_date": None, "end_date": None}

    # Calculate time differences between consecutive measurements
    time_diffs = np.array(
        data.index.diff().total_seconds() / 60
    )  # Convert to minutes

    # Automatically determine dt0 if not provided
    if dt0 is None:
        dt0 = round(np.nanmedian(time_diffs))

    if range_type == "automatic":
        # Determine range of observed data
        min_time = data.index.min()
        max_time = data.index.max()

        # Calculate theoretical number of measurements
        total_minutes = (max_time - min_time).total_seconds() / 60
        theoretical_gl_vals = round(total_minutes / dt0) + 1

        # Calculate missing values due to gaps
        gaps = time_diffs[time_diffs > dt0]
        gap_minutes = gaps.sum()
        n_gaps = len(gaps)
        missing_gl_vals = round((gap_minutes - n_gaps * dt0) / dt0)

        # Calculate number of days
        ndays = (max_time - min_time).total_seconds() / (24 * 3600)

        # Calculate active percentage
        active_percent = (
            (theoretical_gl_vals - missing_gl_vals) / theoretical_gl_vals
        ) * 100
    elif range_type == "manual":
        # Handle consistent end date if provided
        if consistent_end_date is not None:
            end_date = localize_naive_timestamp(pd.to_datetime(consistent_end_date))
        else:
            end_date = data.index.max()
        start_date = end_date - pd.Timedelta(days=int(ndays))

        # Filter data to the specified date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        data = data[mask]

        # Recalculate active percentage for the specified range
        active_percent = (len(data) / (ndays * (24 * (60 / dt0)))) * 100
        min_time = start_date
        max_time = end_date
        ndays = (end_date - start_date).total_seconds() / (24 * 3600)
    else:
        raise ValueError(f"Invalid range_type: {range_type}")

    return {"active_percent": active_percent, "ndays": round(ndays, 1), "start_date": min_time, "end_date": max_time}



