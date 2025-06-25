import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns


def sd_measures(
    data: pd.DataFrame | pd.Series, dt0: Optional[int] = None, inter_gap: int = 45, tz: str = ""
) -> pd.DataFrame | dict[str, float]:
    """
    Calculate SD subtypes for glucose variability analysis

    This function produces SD subtype values in a DataFrame object
    with a row for each subject and columns corresponding to id followed by
    each SD subtype.

    Parameters
    ----------
    data : pd.DataFrame|pd.Series
        DataFrame with columns 'id', 'time', and 'gl' (glucose), or a Series of glucose values
    dt0 : int, optional
        The time frequency for interpolation in minutes
    inter_gap : int, default 45
        The maximum allowable gap (in minutes) for interpolation
    tz : str, default ""
        Timezone specification

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for id and each of the six SD subtypes.
        If a Series of glucose values is passed, then a dictionary is returned.
        - SDw: vertical within days
        - SDhhmm: between time points
        - SDwsh: within series (1-hour windows)
        - SDdm: horizontal sd (between daily means)
        - SDb: between days, within timepoints
        - SDbdm: between days, within timepoints, corrected for daily means

    Details
    -------
    Missing values will be linearly interpolated when close enough to non-missing values.

    SD Subtypes:

    1. SDw - vertical within days:
       Standard deviation of each day's glucose measurements, then mean of all SDs

    2. SDhhmm - between time points:
       Standard deviation of mean glucose values at each time point across days

    3. SDwsh - within series:
       Mean of standard deviations computed over hour-long sliding windows

    4. SDdm - horizontal sd:
       Standard deviation of daily mean glucose values

    5. SDb - between days, within timepoints:
       Mean of standard deviations of glucose values across days for each time point

    6. SDbdm - between days, within timepoints, corrected for changes in daily means:
       Like SDb but after subtracting daily mean from each glucose value

    References
    ----------
    Rodbard (2009) New and Improved Methods to Characterize Glycemic Variability
    Using Continuous Glucose Monitoring. Diabetes Technology and Therapeutics 11, 551-565.

    Examples
    --------
    >>> import pandas as pd
    >>> # Assuming you have glucose data
    >>> result = sd_measures(glucose_data)
    >>> print(result)
    """
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        return sd_measures_single(data, dt0, inter_gap, tz)

    # Data validation (placeholder - implement check_data_columns equivalent)
    data = check_data_columns(data, time_check=True, tz=tz)

    # Convert the dictionary results into a DataFrame with proper columns
    results = []
    for subject_id in data["id"].unique():
        subject_data = data[data["id"] == subject_id]
        result_dict = sd_measures_single(subject_data, dt0, inter_gap, tz)
        result_dict["id"] = subject_id
        results.append(result_dict)

    # convert result into dataframe with 'id' on the first place
    out = pd.DataFrame(results)
    out = out[["id"] + list(out.columns[:-1])]
    return out


def sd_measures_single(
    data: pd.DataFrame, dt0: Optional[int] = None, inter_gap: int = 45, tz: str = ""
) -> dict[str, float]:
    gd2d, actual_dates, gd2d_dt0 = CGMS2DayByDay(data, tz=tz, dt0=dt0, inter_gap=inter_gap)

    return _calculate_sd_subtypes(gd2d, gd2d_dt0)


def _calculate_sd_subtypes(gd2d: np.ndarray, dt0: int) -> Dict[str, Any]:
    """
    Calculate all SD subtypes for a single subject's glucose data matrix

    Parameters
    ----------
    gd2d : np.ndarray
        2D array where rows are days and columns are time points
    dt0 : int
        Time interval in minutes


    Returns
    -------
    dict
        Dictionary containing all SD measures
    """

    result = {}

    # 1. SDw - vertical within days
    # Standard deviation within each day, then mean across days
    daily_sds = _safe_nanstd(gd2d, axis=1, ddof=1)  # ddof=1 for sample std
    result["SDw"] = _safe_nanmean(daily_sds)

    # 2. SDhhmm - between time points
    # Mean at each time point across days, then SD of those means
    timepoint_means = _safe_nanmean(gd2d, axis=0)
    result["SDhhmm"] = _safe_nanstd(timepoint_means, ddof=1)

    # 3. SDwsh - within series (1-hour windows)
    # Rolling standard deviation over 1-hour windows
    win = round(60 / dt0)  # Number of measurements in 1 hour
    gs = gd2d.flatten()  # Vectorize by columns (time-first order)

    # Calculate rolling standard deviation
    rolling_sds = _rolling_std(gs, window=win)
    result["SDwsh"] = _safe_nanmean(rolling_sds)

    # 4. SDdm - horizontal sd (between daily means)
    # Standard deviation of daily mean glucose values
    daily_means = _safe_nanmean(gd2d, axis=1)
    result["SDdm"] = _safe_nanstd(daily_means, ddof=1)

    # 5. SDb - between days, within timepoints
    # SD across days for each time point, then mean of those SDs
    timepoint_sds = _safe_nanstd(gd2d, axis=0, ddof=1)
    result["SDb"] = _safe_nanmean(timepoint_sds)

    # 6. SDbdm - between days, within timepoints, corrected for daily means
    # Subtract daily mean from each value, then calculate SDb on corrected values
    daily_means_matrix = daily_means[:, np.newaxis]  # Convert to column vector
    corrected_gd2d = gd2d - daily_means_matrix
    corrected_timepoint_sds = _safe_nanstd(corrected_gd2d, axis=0, ddof=1)
    result["SDbdm"] = _safe_nanmean(corrected_timepoint_sds)

    return result


def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling standard deviation with non-trimmed ends

    Parameters
    ----------
    data : np.ndarray
        Input data array
    window : int
        Window size for rolling calculation

    Returns
    -------
    np.ndarray
        Rolling standard deviations (trimmed to valid windows only)
    """
    # valid_data = data[~np.isnan(data)]
    valid_data = np.concatenate([data, np.full(window, np.nan)])  # add nan tail to match R
    n = len(valid_data)

    if n < window:
        return np.array([np.nan])

    rolling_stds = []

    for i in range(n - window + 1):
        window_data = valid_data[i : i + window]
        if len(window_data) == window:  # Full window
            rolling_stds.append(_safe_nanstd(window_data, ddof=1))

    return np.array(rolling_stds) if rolling_stds else np.array([np.nan])


def _safe_nanstd(data: np.ndarray, axis: Optional[int] = None, ddof: int = 1) -> float:
    """
    Safe version of np.nanstd that handles insufficient data gracefully

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which the standard deviation is computed
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    float
        Standard deviation or np.nan if insufficient data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if axis is None:
            # Check if we have enough non-NaN values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) <= ddof:
                return np.nan
        else:
            # For axis operations, we need to check each slice
            # This is more complex, so we'll just suppress warnings
            pass

        return np.nanstd(data, axis=axis, ddof=ddof)


def _safe_nanmean(data: np.ndarray, axis: Optional[int] = None) -> float:
    """
    Safe version of np.nanmean that handles empty slices gracefully

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which the mean is computed

    Returns
    -------
    float
        Mean or np.nan if no valid data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if axis is None:
            # Check if we have any non-NaN values
            if np.isnan(data).all():
                return np.nan
        else:
            # For axis operations, suppress warnings and let numpy handle it
            pass

        return np.nanmean(data, axis=axis)


# Alternative vectorized implementation for better performance
def sd_measures_vectorized(
    data: pd.DataFrame, dt0: Optional[int] = None, inter_gap: int = 45, tz: str = ""
) -> pd.DataFrame:
    """
    Vectorized version of sd_measures for better performance with large datasets
    """
    data = check_data_columns(data, time_check=True, tz=tz)

    results = []

    current_dt0 = dt0
    for i, subject_id in enumerate(data["id"].unique()):
        subject_data = data[data["id"] == subject_id].copy()
        gd2d, actual_dates, gd2d_dt0 = CGMS2DayByDay(subject_data, tz=tz, dt0=current_dt0, inter_gap=inter_gap)
        if i == 0:
            current_dt0 = gd2d_dt0
        result = _calculate_sd_subtypes_vectorized(gd2d, gd2d_dt0, subject_id)
        results.append(result)

    return pd.DataFrame(results)


def _calculate_sd_subtypes_vectorized(gd2d: np.ndarray, dt0: int, subject_id: Any) -> Dict[str, Any]:
    """
    Vectorized calculation of SD subtypes using numpy operations
    """
    # Use numpy's built-in functions for better performance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        return {
            "id": subject_id,
            "SDw": _safe_nanmean(np.nanstd(gd2d, axis=1, ddof=1)),
            "SDhhmm": np.nanstd(_safe_nanmean(gd2d, axis=0), ddof=1),
            "SDwsh": _safe_nanmean(_rolling_std(gd2d.T.flatten(), round(60 / dt0))),
            "SDdm": np.nanstd(_safe_nanmean(gd2d, axis=1), ddof=1),
            "SDb": _safe_nanmean(np.nanstd(gd2d, axis=0, ddof=1)),
            "SDbdm": _safe_nanmean(np.nanstd(gd2d - _safe_nanmean(gd2d, axis=1, keepdims=True), axis=0, ddof=1)),
        }
