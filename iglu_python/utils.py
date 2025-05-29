import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
from tzlocal import get_localzone
from datetime import datetime

local_tz = get_localzone() # get the local timezone

def localize_naive_timestamp(timestamp: datetime) -> datetime:
    """
    Localize a naive timestamp to the local timezone.
    """
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(local_tz)
    else:
        return timestamp

def check_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the input DataFrame has the required columns and correct data types.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to check
        
    Returns
    -------
    pd.DataFrame
        Validated DataFrame
        
    Raises
    ------
    ValueError
        If required columns are missing or data types are incorrect
    """
    required_columns = ['id', 'time', 'gl']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(data['gl']):
        try:
            data['gl'] = pd.to_numeric(data['gl'])
        except:
            raise ValueError("Column 'gl' must be numeric")
    
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        try:
            data['time'] = pd.to_datetime(data['time'])
        except:
            raise ValueError("Column 'time' must be datetime")
    
    if not pd.api.types.is_string_dtype(data['id']):
        data['id'] = data['id'].astype(str)
    
    # Check for missing values
    if data['gl'].isna().any():
        warnings.warn("Data contains missing glucose values")
    
    return data

def CGMS2DayByDay(
    data: pd.DataFrame,
    dt0: Optional[pd.Timestamp] = None,
    inter_gap: int = 45,
    tz: str = ""
) -> Tuple[np.ndarray, list, int, list]:
    """
    Interpolate CGM data to a regular time grid.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    dt0 : pd.Timestamp, optional
        Start time for interpolation
    inter_gap : int, default=45
        Maximum gap (in minutes) between measurements to interpolate
    tz : str, default=""
        Time zone to use for calculations
        
    Returns
    -------
    Tuple[np.ndarray, list, int, list]
        - Interpolated glucose values (2D array)
        - List of actual dates
        - Time step in minutes
        - List of gap indices
    """
    # Check data format
    data = check_data_columns(data)
    
    # Convert timezone if specified
    if tz:
        data['time'] = data['time'].dt.tz_localize(tz)
    
    # Get unique subjects
    subjects = data['id'].unique()
    if len(subjects) > 1:
        warnings.warn("Multiple subjects detected. Using first subject only.")
        data = data[data['id'] == subjects[0]]
    
    # Sort by time
    data = data.sort_values('time')
    
    # Calculate time step (dt0)
    if dt0 is None:
        # Use most common time difference
        time_diffs = data['time'].diff().dropna()
        dt0 = int(time_diffs.mode().iloc[0].total_seconds() / 60)
    
    # Create time grid
    start_time = data['time'].min().floor('D')
    end_time = data['time'].max().ceil('D')
    time_grid = pd.date_range(start=start_time, end=end_time, freq=f'{dt0}T')
    
    # Interpolate glucose values
    interp_data = np.interp(
        (time_grid - start_time).total_seconds() / 60,
        (data['time'] - start_time).total_seconds() / 60,
        data['gl']
    )
    
    # Reshape to days
    n_days = (end_time - start_time).days
    n_points_per_day = 24 * 60 // dt0
    interp_data = interp_data.reshape(n_days, n_points_per_day)
    
    # Find gaps
    gaps = []
    for i in range(n_days):
        day_data = interp_data[i]
        # Find indices where gap is larger than inter_gap
        gap_indices = np.where(np.diff(day_data) > inter_gap)[0]
        if len(gap_indices) > 0:
            gaps.append((i, gap_indices))
    
    # Get actual dates
    actual_dates = [start_time + pd.Timedelta(days=i) for i in range(n_days)]
    
    return interp_data, actual_dates, dt0, gaps 