import pandas as pd
import numpy as np
from typing import Tuple, List, Union
from datetime import datetime, timedelta
from .utils import check_data_columns

def cgm2daybyday(data: pd.DataFrame, dt0: Union[int, None] = None, inter_gap: int = 45, tz: str = "") -> Tuple[np.ndarray, List[datetime.date], int]:
    """
    Interpolate glucose values onto an equally spaced grid from day to day.
    
    The function takes CGM data and interpolates it onto a uniform time grid,
    with each row representing a day and each column representing a time point.
    Missing values are linearly interpolated when close enough to non-missing values.
    
    Parameters
    ----------
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
    Tuple[np.ndarray, List[datetime.date], int]
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
    >>> gd2d, dates, dt = cgm2daybyday(data)
    >>> print(gd2d.shape)  # Shape will be (1, 288) for one day with 5-min intervals
    (1, 288)
    """
    # Check and convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', utc=True if tz else False)
        if data['time'].isna().any():
            na_count = data['time'].isna().sum()
            print(f"Warning: During time conversion, {na_count} values were set to NA. Check the correct time zone specification.")
            data = data.dropna(subset=['time'])
    
    # Remove rows with missing values
    data = data.dropna()
    
    # Check for multiple subjects
    unique_ids = data['id'].unique()
    if len(unique_ids) > 1:
        first_id = unique_ids[0]
        print(f"Warning: Data contains more than 1 subject. Only the first subject with id {first_id} is used for output.")
        data = data[data['id'] == first_id]
    
    # Get glucose values
    glucose = data['gl'].values
    
    # Get time data
    times = data['time'].values
    
    # Check for time sorting
    time_diffs = np.diff(times)
    if np.any(time_diffs < 0):
        print(f"Warning: The times for subject {data['id'].iloc[0]} are not in increasing order! The times will be sorted automatically.")
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        glucose = glucose[sort_idx]
        time_diffs = np.diff(times)
    
    # Calculate dt0 if not provided
    if dt0 is None:
        # Convert time differences to minutes and get median
        dt0 = int(np.nanmedian(time_diffs.astype('timedelta64[m]').astype(float)))
    
    # Calculate number of measurements per day
    measurements_per_day = int(24 * 60 / dt0)
    
    # Get date range
    min_date = pd.Timestamp(times[0]).date()
    max_date = pd.Timestamp(times[-1]).date()
    date_range = pd.date_range(min_date, max_date, freq='D')
    ndays = len(date_range)
    
    # Create time grid for one day
    day_start = pd.Timestamp(min_date)
    time_grid = pd.date_range(day_start, day_start + timedelta(days=1), freq=f'{dt0}min', inclusive='left')
    time_grid = time_grid.time
    
    # Initialize output matrix
    gd2d = np.full((ndays, measurements_per_day), np.nan)
    
    # Process each day
    for i, date in enumerate(date_range):
        # Get data for this day
        day_mask = np.array([pd.Timestamp(t).date() == date for t in times])
        day_times = times[day_mask]
        day_glucose = glucose[day_mask]
        
        if len(day_times) == 0:
            continue
        
        # Convert times to minutes since midnight
        day_times_minutes = np.array([t.hour * 60 + t.minute for t in pd.Timestamp(day_times).time])
        grid_minutes = np.array([t.hour * 60 + t.minute for t in time_grid])
        
        # Find indices for interpolation
        valid_indices = ~np.isnan(day_glucose)
        if np.sum(valid_indices) < 2:
            continue
        
        # Interpolate glucose values
        interpolated = np.interp(grid_minutes, 
                               day_times_minutes[valid_indices],
                               day_glucose[valid_indices],
                               left=np.nan, right=np.nan)
        
        # Check for gaps larger than inter_gap
        if len(day_times) > 1:
            gaps = np.diff(day_times_minutes)
            gap_indices = np.where(gaps > inter_gap)[0]
            
            for gap_idx in gap_indices:
                gap_start = day_times_minutes[gap_idx]
                gap_end = day_times_minutes[gap_idx + 1]
                mask = (grid_minutes > gap_start) & (grid_minutes < gap_end)
                interpolated[mask] = np.nan
        
        gd2d[i, :] = interpolated
    
    return gd2d, [d.date() for d in date_range], dt0 