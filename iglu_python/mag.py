from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns, IGLU_R_COMPATIBLE


def mag(
    data: Union[pd.DataFrame, pd.Series],
    n: int = 60,
    dt0: Optional[int] = None,
    inter_gap: int = 45,
    tz: str = "",
) -> pd.DataFrame:
    """
    Calculate Mean Absolute Glucose (MAG).

    The function calculates the mean absolute glucose change over specified time intervals.
    The glucose values are linearly interpolated over a time grid starting at the beginning
    of the first day of data and ending on the last day of data. Then, MAG is calculated as
    |ΔG|/Δt where |ΔG| is the sum of the absolute change in glucose calculated for each
    interval as specified by n, and Δt is the total time in hours.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    n : int, default=60
        Integer giving the desired interval in minutes over which to calculate
        the change in glucose. Default is 60 for hourly intervals.
    dt0 : Optional[int], default=None
        Time interval between measurements in minutes. If None, it will be automatically
        determined from the data.
    inter_gap : int, default=45
        Maximum gap in minutes for interpolation. Gaps larger than this will not be
        interpolated.
    tz : str, default=""
        Time zone to use for datetime conversion. Empty string means use local time zone.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - MAG: Mean Absolute Glucose value

    References
    ----------
    Hermanides et al. (2010) Glucose Variability is Associated with Intensive Care Unit
    Mortality, Critical Care Medicine 38(3) 838-842,
    doi:10.1097/CCM.0b013e3181cc4be9

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> mag(data)
       id    MAG
    0  subject1  60.0
    1  subject2  72.0

    >>> mag(data['gl'], n=30)
       MAG
    0  66.0
    """

    def mag_single(data: pd.DataFrame, n: int) -> float:
        """Calculate MAG for a single subject"""
        # Convert data to day-by-day format
        data_ip = CGMS2DayByDay(data, dt0=dt0, inter_gap=inter_gap, tz=tz)
        dt0_actual = data_ip[2]  # Time between measurements in minutes

        # Ensure n is not less than data collection frequency
        if n < dt0_actual:
            n = dt0_actual

        # Calculate number of readings per interval
        readings_per_interval = round(n / dt0_actual)

        # Get glucose values and calculate differences
        gl_values = data_ip[0].flatten()  # Flatten the matrix
        # gl_values = gl_values[~np.isnan(gl_values)]  # Remove NaN values

        if len(gl_values) <= 1:
            return 0.0

        # Calculate absolute differences between readings n minutes apart
        lag = readings_per_interval

        if IGLU_R_COMPATIBLE:
            idx = np.arange(0,len(gl_values),lag)
            gl_values_idx = gl_values[idx]
            diffs = gl_values_idx[1:] - gl_values_idx[:-1]
            diffs = np.abs(diffs)
            diffs = diffs[~np.isnan(diffs)]
            # to be IGLU-R test compatible, imho they made error.
            # has to be total_time_hours = ((len(diffs)) * n) / 60   
            total_time_hours = ((len(gl_values_idx[~np.isnan(gl_values_idx)])) * n) / 60
            if total_time_hours == 0:
                return 0.0
            mag = float(np.sum(diffs) / total_time_hours)
        else:
            diffs = gl_values[lag:] - gl_values[:-lag]
            diffs = np.abs(diffs)
            diffs = diffs[~np.isnan(diffs)]

            # Calculate MAG: sum of absolute differences divided by total time in hours
            total_time_hours = ((len(diffs)) * n) / 60   
            if total_time_hours == 0:
                return 0.0
            mag = float(np.sum(diffs) / total_time_hours)

        return mag

    # Handle Series input
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame format
        data_df = pd.DataFrame(
            {
                "id": ["subject1"] * len(data),
                "time": pd.date_range(
                    start="2020-01-01", periods=len(data), freq="5min"
                ),
                "gl": data.values,
            }
        )
        mag_val = mag_single(data_df, n)
        return pd.DataFrame({"MAG": [mag_val]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Ensure n is an integer
    if not isinstance(n, int):
        n = round(n)

    # Calculate MAG for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].copy()
        if len(subject_data.dropna(subset=["gl"])) == 0:
            continue

        mag_val = mag_single(subject_data, n)
        result.append({"id": subject, "MAG": mag_val})

    return pd.DataFrame(result)
