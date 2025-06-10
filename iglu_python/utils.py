import warnings
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tzlocal import get_localzone

local_tz = get_localzone()  # get the local timezone

IGLU_R_COMPATIBLE = True

def localize_naive_timestamp(timestamp: datetime) -> datetime:
    """
    Localize a naive timestamp to the local timezone.
    """
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(local_tz)
    else:
        return timestamp


def set_local_tz(tz: str) -> None:
    """
    Set the local timezone.
    It used ONLY in the unittests to fix configuration as in expected results.
    """
    global local_tz
    local_tz = ZoneInfo(tz)

def get_local_tz() :
    global local_tz
    return local_tz


def check_data_columns(data: pd.DataFrame, time_check=False, tz="") -> pd.DataFrame:
    """
    Check if the input DataFrame has the required columns and correct data types.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to check
    tz : str, default=""
        Time zone to use for calculations
        If tz is not "", the time column is converted to the specified timezone

    Returns
    -------
    pd.DataFrame
        Validated DataFrame

    Raises
    ------
    ValueError
        If required columns are missing or data types are incorrect
    """
    required_columns = ["id", "time", "gl"]

    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create a copy to avoid dtype warning
    data = data.copy()

    # Check data types
    if not pd.api.types.is_numeric_dtype(data["gl"]):
        try:
            data["gl"] = pd.to_numeric(data["gl"])
        except:
            raise ValueError("Column 'gl' must be numeric")

    if not pd.api.types.is_datetime64_any_dtype(data["time"]):
        try:
            data["time"] = pd.to_datetime(data["time"])
        except:
            raise ValueError("Column 'time' must be datetime")

    if not pd.api.types.is_string_dtype(data["id"]):
        data["id"] = data["id"].astype(str)

    # check if data frame empty
    if data.empty:
        raise ValueError("Data frame is empty")

    # Check if data contains no glucose values
    if data["gl"].isna().all():
        raise ValueError("Data contains no glucose values")

    # Check for missing values
    if data["gl"].isna().any():
        warnings.warn("Data contains missing glucose values")

    # convert time to specified timezone
    # TODO: check if this is correct (R-implementation compatibility)
    # if tz and tz != "":
    #     # First remove timezone information, then localize to specified timezone
    #     data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None).dt.tz_localize(tz)
    #
    # this is implementation compatible with R implementation
    # but seems incorrect, as it convert time to TZ instead of localizing it to TZ
    if tz != "":
        # Create a copy to avoid dtype warning and properly handle timezone conversion
        data["time"] = pd.to_datetime(data["time"]).apply(localize_naive_timestamp).dt.tz_convert(tz)
    else:
        # Create a copy to avoid dtype warning
        data["time"] = pd.to_datetime(data["time"]).apply(localize_naive_timestamp)

    return data


def CGMS2DayByDay(
    data: pd.DataFrame,
    dt0: Optional[pd.Timestamp] = None,
    inter_gap: int = 45,
    tz: str = "",
) -> Tuple[np.ndarray, list, int]:
    """
    Interpolate glucose values onto an equally spaced grid from day to day.

    The function takes CGM data and interpolates it onto a uniform time grid,
    with each row representing a day and each column representing a time point.
    Missing values are linearly interpolated when close enough to non-missing values.

    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'. Should only be data for 1 subject.
        In case multiple subject ids are detected, a warning is produced and only 1st subject is used.
    dt0 : int, optional
        The time frequency for interpolation in minutes. If None, will match the CGM meter's frequency
        (e.g., 5 min for Dexcom).
    inter_gap : int, default=45
        The maximum allowable gap (in minutes) for interpolation. Values will not be interpolated
        between glucose measurements that are more than inter_gap minutes apart.
    tz : str, default=""
        Time zone to use for datetime conversion. Empty string means use local time zone.

    Returns
    -------
    Tuple[np.ndarray, list, int, list]
        A tuple containing:
        - gd2d: A 2D numpy array of glucose values with each row corresponding to a new day,
               and each column corresponding to time
        - actual_dates: A list of dates corresponding to the rows of gd2d
        - dt0: Time frequency of the resulting grid, in minutes
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1'] * 4,
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
    ...     'gl': [150, 200, 180, 160]
    ... })
    >>> gd2d, dates, dt = CGMS2DayByDay(data)
    >>> print(gd2d.shape)  # Shape will be (1, 288) for one day with 5-min intervals
    (1, 288)
    """
    # Check data format
    data = check_data_columns(data, tz)

    # Get unique subjects
    subjects = data["id"].unique()
    if len(subjects) > 1:
        raise ValueError("Multiple subjects detected. Please provide a single subject.")

    # Sort by time
    data = data.sort_values("time")

    # Calculate time step (dt0)
    if dt0 is None:
        # Use most common time difference
        time_diffs = data["time"].diff().dropna()
        dt0 = int(time_diffs.mode().iloc[0].total_seconds() / 60)

    # Create time grid
    start_time = data["time"].min().floor("D")
    end_time = data["time"].max().ceil("D")
    time_grid = pd.date_range(
        start=start_time, end=end_time, freq=f"{dt0}min"
    )
    if IGLU_R_COMPATIBLE:
        # remove the first time point
        time_grid = time_grid[1:]
    else:
        # remove the last time point
        time_grid = time_grid[:-1]

    # find gaps in the data (using original data indexes, not time grid)
    gaps = []
    for i in range(len(data) - 1):
        if (
            data["time"].iloc[i + 1] - data["time"].iloc[i]
        ).total_seconds() > inter_gap * 60:
            gaps.append((i, i + 1))

    # Interpolate glucose values
    interp_data = np.interp(
        (time_grid - start_time).total_seconds() / 60,
        (data["time"] - start_time).dt.total_seconds() / 60,
        data["gl"],
        left=np.nan,
        right=np.nan,
    )

    # put nan in the gaps
    for gap in gaps:
        gap_start_idx = gap[0]
        gap_start_time = data["time"].iloc[gap_start_idx]
        # find the index of the gap start in the time grid
        gap_start_idx_in_time_grid = int(
            np.floor((gap_start_time - start_time).total_seconds() / (60 * dt0))
        )
        gap_end_idx = gap[1]
        gap_end_time = data["time"].iloc[gap_end_idx]
        # find the index of the gap end in the time grid
        gap_end_idx_in_time_grid = int(
            np.floor(((gap_end_time - start_time).total_seconds() -1 ) / (60 * dt0)) # -1sec to indicate time before measurement
        )
        # put nan in the gap
        interp_data[gap_start_idx_in_time_grid:gap_end_idx_in_time_grid] = np.nan

    # for compatibility with the R package, set values to nan before data['time'].min() and after data['time'].max()
    # find index of timegrid before data['time'].min() and after data['time'].max()
    # head_min_idx = np.where(time_grid >= data['time'].min())[0][0]
    # tail_max_idx = np.where(time_grid <= data['time'].max())[0][-1] + 1
    # interp_data[:head_min_idx] = np.nan
    # interp_data[tail_max_idx:] = np.nan

    # Reshape to days
    n_days = (end_time - start_time).days
    n_points_per_day = 24 * 60 // dt0
    interp_data = interp_data.reshape(n_days, n_points_per_day)

    # Get actual dates
    if IGLU_R_COMPATIBLE:
        # convert start_time into naive datetime
        start_time = start_time.tz_localize(None)
        
    actual_dates = [start_time + pd.Timedelta(days=i) for i in range(n_days)]

    return interp_data, actual_dates, dt0

def gd2d_to_df(gd2d, actual_dates, dt0):
    """Convert gd2d (CGMS2DayByDay output) to a pandas DataFrame"""
    df = pd.DataFrame({"time": [], "gl": []})

    gl = gd2d.flatten().tolist()
    time = []
    for day in range(gd2d.shape[0]):
        n = len(gd2d[day, :])
        day_time = [pd.Timedelta(i * dt0, unit="m") + actual_dates[day] for i in range(n)]
        time.extend(day_time)

    df = pd.DataFrame({
            "time": pd.Series(time, dtype='datetime64[ns]'),
            "gl": pd.Series(gl, dtype='float64')
        })

    return df
