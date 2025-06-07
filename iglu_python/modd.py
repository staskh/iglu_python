from typing import Union

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns


def modd(
    data: Union[pd.DataFrame, pd.Series], lag: int = 1, tz: str = ""
) -> pd.DataFrame:
    """
    Calculate Mean of Daily Differences (MODD).

    The function calculates MODD values by taking the mean of absolute differences between
    glucose measurements at the same time of day, with an optional lag parameter to compare
    values that are multiple days apart.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    lag : int, default=1
        Integer indicating which lag (# days) to use. Default is 1.
    tz : str, default=""
        Time zone to use for datetime conversion. Empty string means use local time zone.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - MODD: Mean of Daily Differences value

    References
    ----------
    Service, F. J. & Nelson, R. L. (1980) Characteristics of glycemic stability.
    Diabetes care 3:58-62, doi:10.2337/diacare.3.1.58.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> modd(data)
       id    MODD
    0  subject1  50.0
    1  subject2  60.0

    >>> modd(data['gl'], lag=2)
       MODD
    0  45.0
    """

    def modd_single(data: pd.DataFrame) -> float:
        """Calculate MODD for a single subject"""
        # Convert data to day-by-day format
        data_ip = CGMS2DayByDay(data, tz=tz)
        gl_by_id_ip = data_ip[0].flatten()  # Get interpolated glucose values
        dt0 = data_ip[2]  # Get time frequency

        # Calculate absolute differences with specified lag
        # lag is in days, so we need to convert to minutes and divide of dt0 frequency
        shift = int(lag * 24 * 60 / dt0)  # Convert lag to minutes and divide by dt0
        # Shift array by lag and calculate differences
        abs_diffs = np.abs(gl_by_id_ip[shift:] - gl_by_id_ip[:-shift])
        # Remove NaNs
        abs_diffs = abs_diffs[~np.isnan(abs_diffs)]  # Remove NaNs

        # Calculate mean of absolute differences, ignoring NaN values
        modd_val = np.nanmean(abs_diffs)

        return float(modd_val) if not pd.isna(modd_val) else np.nan

    # Handle Series input
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        data_df = pd.DataFrame(
            {
                "id": ["subject1"] * len(data.values),
                "time": data.index,
                "gl": data.values,
            }
        )

        modd_val = modd_single(data_df)
        return pd.DataFrame({"MODD": [modd_val]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate MODD for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].copy()
        if len(subject_data.dropna(subset=["gl"])) == 0:
            continue

        modd_val = modd_single(subject_data)
        result.append({"id": subject, "MODD": modd_val})

    return pd.DataFrame(result)
