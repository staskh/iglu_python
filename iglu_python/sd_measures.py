import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import warnings

from .utils import check_data_columns,CGMS2DayByDay 

def sd_measures(data: pd.DataFrame, 
                dt0: Optional[int] = None, 
                inter_gap: int = 45, 
                tz: str = "") -> pd.DataFrame:
    """
    Calculate SD subtypes for glucose variability analysis
    
    This function produces SD subtype values in a DataFrame object
    with a row for each subject and columns corresponding to id followed by
    each SD subtype.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl' (glucose)
    dt0 : int, optional
        The time frequency for interpolation in minutes
    inter_gap : int, default 45
        The maximum allowable gap (in minutes) for interpolation
    tz : str, default ""
        Timezone specification
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for id and each of the six SD subtypes:
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
    
    # Data validation (placeholder - implement check_data_columns equivalent)
    data = check_data_columns(data, time_check=True, tz=tz)
    
    subjects = data['id'].unique()
    n_subjects = len(subjects)
    
    # Calculate uniform grid for all subjects
    gdall = []
    current_dt0 = dt0
    
    for i, subject_id in enumerate(subjects):
        subject_data = data[data['id'] == subject_id].copy()
        
        # Convert to day-by-day format (placeholder - implement CGMS2DayByDay equivalent)
        gd2d, actual_dates, gd2d_dt0 = CGMS2DayByDay(subject_data, tz=tz, dt0=current_dt0, inter_gap=inter_gap)
        gdall.append(gd2d)
        
        # Use the dt0 from first subject for consistency
        if i == 0:
            current_dt0 = gd2d_dt0
    
    dt0 = current_dt0
    
    # Calculate SD measures for each subject
    results = []
    
    for i, gd2d in enumerate(gdall):
        subject_id = subjects[i]
        result = _calculate_sd_subtypes(gd2d, dt0, subject_id)
        results.append(result)
    
    # Combine results
    final_results = pd.DataFrame(results)
    
    return final_results


def _calculate_sd_subtypes(gd2d: np.ndarray, dt0: int, subject_id: Any) -> Dict[str, Any]:
    """
    Calculate all SD subtypes for a single subject's glucose data matrix
    
    Parameters
    ----------
    gd2d : np.ndarray
        2D array where rows are days and columns are time points
    dt0 : int
        Time interval in minutes
    subject_id : Any
        Subject identifier
        
    Returns
    -------
    dict
        Dictionary containing all SD measures
    """
    
    result = {'id': subject_id}
    
    # 1. SDw - vertical within days
    # Standard deviation within each day, then mean across days
    daily_sds = np.nanstd(gd2d, axis=1, ddof=1)  # ddof=1 for sample std
    result['SDw'] = np.nanmean(daily_sds)
    
    # 2. SDhhmm - between time points
    # Mean at each time point across days, then SD of those means
    timepoint_means = np.nanmean(gd2d, axis=0)
    result['SDhhmm'] = np.nanstd(timepoint_means, ddof=1)
    
    # 3. SDwsh - within series (1-hour windows)
    # Rolling standard deviation over 1-hour windows
    win = round(60 / dt0)  # Number of measurements in 1 hour
    gs = gd2d.flatten()  # Vectorize by columns (time-first order)
    
    # Calculate rolling standard deviation
    rolling_sds = _rolling_std(gs, window=win)
    result['SDwsh'] = np.nanmean(rolling_sds)
    
    # 4. SDdm - horizontal sd (between daily means)
    # Standard deviation of daily mean glucose values
    daily_means = np.nanmean(gd2d, axis=1)
    result['SDdm'] = np.nanstd(daily_means, ddof=1)
    
    # 5. SDb - between days, within timepoints
    # SD across days for each time point, then mean of those SDs
    timepoint_sds = np.nanstd(gd2d, axis=0, ddof=1)
    result['SDb'] = np.nanmean(timepoint_sds)
    
    # 6. SDbdm - between days, within timepoints, corrected for daily means
    # Subtract daily mean from each value, then calculate SDb on corrected values
    daily_means_matrix = daily_means[:, np.newaxis]  # Convert to column vector
    corrected_gd2d = gd2d - daily_means_matrix
    corrected_timepoint_sds = np.nanstd(corrected_gd2d, axis=0, ddof=1)
    result['SDbdm'] = np.nanmean(corrected_timepoint_sds)
    
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
    #valid_data = data[~np.isnan(data)]
    valid_data = np.concatenate([data, np.full(window, np.nan)])  # add nan tail to match R
    n = len(valid_data)
    
    if n < window:
        return np.array([np.nan])
    
    rolling_stds = []
    
    for i in range(n - window + 1):
        window_data = valid_data[i:i + window]
        if len(window_data) == window:  # Full window
            rolling_stds.append(np.nanstd(window_data, ddof=1))
    
    return np.array(rolling_stds) if rolling_stds else np.array([np.nan])


# Alternative vectorized implementation for better performance
def sd_measures_vectorized(data: pd.DataFrame, 
                          dt0: Optional[int] = None, 
                          inter_gap: int = 45, 
                          tz: str = "") -> pd.DataFrame:
    """
    Vectorized version of sd_measures for better performance with large datasets
    """
    data = check_data_columns(data, time_check=True, tz=tz)
    
    results = []
    
    for subject_id in data['id'].unique():
        subject_data = data[data['id'] == subject_id].copy()
        gd2d, actual_dates, gd2d_dt0 = CGMS2DayByDay(subject_data, tz=tz, dt0=current_dt0, inter_gap=inter_gap)
        
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
            'id': subject_id,
            'SDw': np.nanmean(np.nanstd(gd2d, axis=1, ddof=1)),
            'SDhhmm': np.nanstd(np.nanmean(gd2d, axis=0), ddof=1),
            'SDwsh': np.nanmean(_rolling_std(gd2d.T.flatten(), round(60/dt0))),
            'SDdm': np.nanstd(np.nanmean(gd2d, axis=1), ddof=1),
            'SDb': np.nanmean(np.nanstd(gd2d, axis=0, ddof=1)),
            'SDbdm': np.nanmean(np.nanstd(gd2d - np.nanmean(gd2d, axis=1, keepdims=True), 
                                        axis=0, ddof=1))
        }
