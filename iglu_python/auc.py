import pandas as pd
import numpy as np
from typing import Optional, Union
from datetime import datetime
from .utils import check_data_columns
from .utils import CGMS2DayByDay

def auc(data: pd.DataFrame, tz: str = "") -> pd.DataFrame:
    """
    Calculate Area Under Curve (AUC) for glucose measurements.
    
    The function produces hourly average AUC for each subject. AUC is calculated 
    for every hour using the trapezoidal rule, then hourly average AUC is calculated 
    for each 24 hour period, then the mean of hourly average AUC across all 24 hour 
    periods is returned as overall hourly average AUC.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    tz : str, default=""
        Time zone to be used. Empty string means current time zone, "GMT" means UTC.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier
        - hourly_auc: hourly average AUC value
        
    References
    ----------
    Danne et al. (2017) International Consensus on Use of Continuous Glucose Monitoring,
    Diabetes Care 40:1631-1640,
    doi:10.2337/dc17-1600.
        
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:10:00', '2020-01-01 00:00:00',
    ...                            '2020-01-01 00:05:00']),
    ...     'gl': [150, 155, 160, 140, 145]
    ... })
    >>> auc(data)
       id  hourly_auc
    0  subject1      155.0
    1  subject2      142.5
    """
    # Check data format and convert time to datetime
    data = check_data_columns(data)
    
    def auc_single(subject_data: pd.DataFrame) -> float:
        """Calculate AUC for a single subject"""
        # Get interpolated data using CGMS2DayByDay
        gd2d,actual_dates,dt0 = CGMS2DayByDay(subject_data, tz=tz)
        
        # Create DataFrame with day and glucose values
        days = np.repeat(actual_dates, 1440/dt0)
        gl_values = gd2d.flatten()
        
        # Group by day and calculate AUC for each day
        daily_data = pd.DataFrame({
            'day': days,
            'gl': gl_values
        })
        
        # Calculate AUC for each day using trapezoidal rule
        daily_auc = daily_data.groupby('day').apply(
            lambda x:  np.nansum((x['gl'].iloc[1:].values + x['gl'].iloc[:-1].values) / 2)
        )
        
        # number of trapezoidal areas
        n_trapezoids = daily_data.groupby('day').apply(
            lambda x: (len(x['gl'].dropna()) - 1)
        )
        
        # Calculate hourly average AUC for each day
        hourly_avg = daily_auc / n_trapezoids
        
        # Return mean of daily hourly averages
        return hourly_avg.mean()
    
    # Process each subject
    result = []
    for subject in data['id'].unique():
        subject_data = data[data['id'] == subject]
        hourly_auc = auc_single(subject_data)
        result.append({
            'id': subject,
            'hourly_auc': hourly_auc
        })
    
    # Convert to DataFrame
    return pd.DataFrame(result) 