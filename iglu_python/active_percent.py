from datetime import datetime
from typing import Optional, Union

import pandas as pd

from .utils import check_data_columns, localize_naive_timestamp


def active_percent(
    data: pd.DataFrame,
    dt0: Optional[int] = None,
    tz: str = "",
    range_type: str = "automatic",
    ndays: int = 14,
    consistent_end_date: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
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

        if len(sub_data) == 0:
            continue

        # Calculate time differences between consecutive measurements
        time_diffs = (
            sub_data["time"].diff().dt.total_seconds() / 60
        )  # Convert to minutes

        # Automatically determine dt0 if not provided
        if dt0 is None:
            dt0 = round(time_diffs.median())

        if range_type == "automatic":
            # Determine range of observed data
            min_time = sub_data["time"].min()
            max_time = sub_data["time"].max()

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
                end_date = sub_data["time"].max()
            start_date = end_date - pd.Timedelta(days=int(ndays))

            # Filter data to the specified date range
            mask = (sub_data["time"] >= start_date) & (sub_data["time"] <= end_date)
            sub_data = sub_data[mask]

            # Recalculate active percentage for the specified range
            active_percent = (len(sub_data) / (ndays * (24 * (60 / dt0)))) * 100
            min_time = start_date
            max_time = end_date
            ndays = (end_date - start_date).total_seconds() / (24 * 3600)
        else:
            raise ValueError(f"Invalid range_type: {range_type}")

        active_perc_data.append(
            {
                "id": subject,
                "active_percent": active_percent,
                "ndays": round(ndays, 1),
                "start_date": min_time,
                "end_date": max_time,
            }
        )

    # Convert to DataFrame
    result = pd.DataFrame(active_perc_data)

    # If input was a Series (glucose values only), remove id column
    if hasattr(data, "is_vector") and data.is_vector:
        result = result.drop("id", axis=1)

    return result
