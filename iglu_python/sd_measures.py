from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage

from .utils import check_data_columns, CGMS2DayByDay


def sd_measures(
    data: pd.DataFrame, 
    dt0: Optional[int] = None, 
    inter_gap: int = 45, 
    tz: str = ""
) -> pd.DataFrame:
    """
    Calculate SD subtypes for glucose variability analysis.

    The function produces SD subtype values in a DataFrame with a row for each 
    subject and columns corresponding to id followed by each SD subtype.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'. Should only be data for 
        1 subject. In case multiple subject ids are detected, an error is raised.
    dt0 : int, optional
        The time frequency for interpolation in minutes. If None, will match 
        the CGM meter's frequency (e.g., 5 min for Dexcom).
    inter_gap : int, default=45
        The maximum allowable gap (in minutes) for interpolation. Values will 
        not be interpolated between glucose measurements that are more than 
        inter_gap minutes apart.
    tz : str, default=""
        Time zone to use for calculations. If tz is not "", the time column 
        is converted to the specified timezone.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and 
        a column for each of the six SD subtypes:
        - SDw: vertical within days
        - SDhhmm: between time points  
        - SDwsh: within series (1-hour windows)
        - SDdm: horizontal sd (daily means)
        - SDb: between days, within timepoints
        - SDbdm: between days, within timepoints, corrected for daily means

    Raises
    ------
    ValueError
        If multiple subjects are detected in the data.

    Notes
    -----
    Missing values will be linearly interpolated when close enough to non-missing values.

    The six SD subtypes are:

    1. **SDw - vertical within days:**
       Calculated by first taking the standard deviation of each day's glucose 
       measurements, then taking the mean of all the standard deviations.

    2. **SDhhmm - between time points:**
       Also known as SDhh:mm. Calculated by taking the mean glucose values at 
       each time point in the grid across days, and taking the standard deviation 
       of those means.

    3. **SDwsh - within series:**
       Also known as SDws h. Calculated by taking the hour-long intervals starting 
       at every point in the interpolated grid, computing the standard deviation 
       of the points in each hour-long interval, and then finding the mean of 
       those standard deviations.

    4. **SDdm - horizontal sd:**
       Calculated by taking the daily mean glucose values, and then taking the 
       standard deviation of those daily means.

    5. **SDb - between days, within timepoints:**
       Calculated by taking the standard deviation of the glucose values across 
       days for each time point, and then taking the mean of those standard deviations.

    6. **SDbdm - between days, within timepoints, corrected for changes in daily means:**
       Also known as SDb // dm. Calculated by subtracting the daily mean from 
       each glucose value, then taking the standard deviation of the corrected 
       glucose values across days for each time point, and then taking the mean 
       of those standard deviations.

    References
    ----------
    Rodbard (2009) New and Improved Methods to Characterize Glycemic Variability
    Using Continuous Glucose Monitoring. Diabetes Technology and Therapeutics 11,
    551-565, doi:10.1089/dia.2009.0015.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1'] * 4,
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
    ...     'gl': [150, 200, 180, 160]
    ... })
    >>> result = sd_measures(data)
    >>> print(result.columns)
    Index(['id', 'SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm'], dtype='object')
    """
    # Check data format
    data = check_data_columns(data, tz)
    
    # Get unique subjects
    subjects = data["id"].unique()
    if len(subjects) > 1:
        raise ValueError("Multiple subjects detected. Please provide a single subject.")
    
    subject = subjects[0]
    
    # Sort by time
    data = data.sort_values("time")
    
    # Calculate uniform grid using CGMS2DayByDay
    gd2d, actual_dates, dt0 = CGMS2DayByDay(data, dt0=dt0, inter_gap=inter_gap, tz=tz)
    
    # Calculate SD measures
    results = {}
    results['id'] = subject
    
    # SDw - vertical within days
    # Standard deviation of each day's glucose measurements, then mean of all SDs
    daily_sds = np.nanstd(gd2d, axis=1, ddof=1)  # axis=1 for row-wise (daily) SD
    results['SDw'] = np.nanmean(daily_sds)
    
    # SDhhmm - between time points
    # Mean glucose at each time point across days, then SD of those means
    timepoint_means = np.nanmean(gd2d, axis=0)  # axis=0 for column-wise (timepoint) means
    results['SDhhmm'] = np.nanstd(timepoint_means, ddof=1)
    
    # SDwsh - within series - for 1 hour window
    # Calculate rolling standard deviation with 1-hour windows
    win = round(60 / dt0)  # how many measurements are within 1 hour
    gs = gd2d.T.flatten()  # flatten row by row (transpose then flatten)
    
    # Calculate rolling standard deviation using pandas rolling with min_periods
    gs_series = pd.Series(gs)
    rolling_sds = gs_series.rolling(window=win, min_periods=1, center=False).std()
    # Remove NaN values and calculate mean
    valid_rolling_sds = rolling_sds.dropna()
    results['SDwsh'] = np.nanmean(valid_rolling_sds)
    
    # SDdm - "Horizontal" sd
    # Standard deviation of daily means
    daily_means = np.nanmean(gd2d, axis=1)  # axis=1 for row-wise (daily) means
    results['SDdm'] = np.nanstd(daily_means, ddof=1)
    
    # SDb - between days, within timepoints
    # Standard deviation across days for each time point, then mean of those SDs
    timepoint_sds = np.nanstd(gd2d, axis=0, ddof=1)  # axis=0 for column-wise (timepoint) SDs
    results['SDb'] = np.nanmean(timepoint_sds)
    
    # SDbdm - between days, within timepoints, corrected for changes in daily means
    # Subtract daily mean from each value, then calculate SDs across days for each timepoint
    daily_means_matrix = daily_means[:, np.newaxis]  # Convert to column vector for broadcasting
    corrected_gd2d = gd2d - daily_means_matrix
    corrected_timepoint_sds = np.nanstd(corrected_gd2d, axis=0, ddof=1)
    results['SDbdm'] = np.nanmean(corrected_timepoint_sds)
    
    # Create result DataFrame
    result_df = pd.DataFrame([results])
    
    return result_df 