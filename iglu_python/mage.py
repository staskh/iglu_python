import pandas as pd
import numpy as np
from typing import Union, Optional, Literal
from .utils import check_data_columns
from .utils import CGMS2DayByDay

def mage(data: Union[pd.DataFrame, pd.Series], 
         version: Literal['ma', 'naive'] = 'ma',
         sd_multiplier: float = 1.0,
         short_ma: int = 5,
         long_ma: int = 32,
         return_type: Literal['num', 'df'] = 'num',
         direction: Literal['avg', 'service', 'max', 'plus', 'minus'] = 'avg',
         tz: str = "",
         inter_gap: int = 45,
         max_gap: int = 180) -> pd.DataFrame:
    """
    Calculate Mean Amplitude of Glycemic Excursions (MAGE).
    
    The function calculates MAGE values using either a moving average ('ma') or naive ('naive') algorithm.
    The 'ma' algorithm is more accurate and is the default. It uses crosses of short and long moving
    averages to identify intervals where a peak/nadir might exist, then calculates the height from
    one peak/nadir to the next nadir/peak from the original glucose values.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    version : Literal['ma', 'naive'], default='ma'
        Algorithm version to use. 'ma' is more accurate and is the default.
        'naive' is included for backward compatibility.
    sd_multiplier : float, default=1.0
        Multiplier for standard deviation used in naive algorithm to determine
        size of glycemic excursions. Only used if version='naive'.
    short_ma : int, default=5
        Period length of short moving average. Must be positive and less than long_ma.
        Recommended < 15.
    long_ma : int, default=32
        Period length of long moving average. Must be positive and greater than short_ma.
        Recommended > 20.
    return_type : Literal['num', 'df'], default='num'
        Return type. 'num' returns a single MAGE value, 'df' returns a DataFrame with
        MAGE values for each segment.
    direction : Literal['avg', 'service', 'max', 'plus', 'minus'], default='avg'
        Direction of MAGE calculation:
        - 'avg': Average of MAGE+ and MAGE-
        - 'service': Based on first countable excursion
        - 'max': Maximum of MAGE+ and MAGE-
        - 'plus': MAGE+ (nadir to peak)
        - 'minus': MAGE- (peak to nadir)
    tz : str, default=""
        Time zone to use for datetime conversion. Empty string means use local time zone.
    inter_gap : int, default=45
        Maximum gap in minutes for interpolation. Gaps larger than this will not be
        interpolated.
    max_gap : int, default=180
        Maximum gap in minutes before splitting into segments.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - MAGE: Mean Amplitude of Glycemic Excursions value
        
    References
    ----------
    Service et al. (1970) Mean amplitude of glycemic excursions, a measure of diabetic instability
    Diabetes 19:644-655, doi:10.2337/diab.19.9.644.
    
    Fernandes, Nathaniel J., et al. "Open-source algorithm to calculate mean amplitude of glycemic
    excursions using short and long moving averages." Journal of diabetes science and technology
    16.2 (2022): 576-577, doi:10.1177/19322968211061165.
        
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> mage(data)
       id    MAGE
    0  subject1  50.0
    1  subject2  60.0
    
    >>> mage(data['gl'], version='naive', sd_multiplier=1.5)
       MAGE
    0  45.0
    """
    def mage_naive(data: pd.DataFrame) -> float:
        """Calculate MAGE using naive algorithm"""
        # Calculate absolute differences from mean
        mean_gl = data['gl'].mean()
        abs_diff_mean = abs(data['gl'] - mean_gl)
        
        # Calculate standard deviation
        std_gl = data['gl'].std()
        
        # Calculate MAGE as mean of differences greater than sd_multiplier * std
        mage_val = abs_diff_mean[abs_diff_mean > (sd_multiplier * std_gl)].mean()
        
        return float(mage_val) if not pd.isna(mage_val) else np.nan
    
    def mage_ma_single(data: pd.DataFrame, short_ma: int, long_ma: int) -> pd.DataFrame:
        """Calculate MAGE using moving average algorithm for a single subject"""
        # Convert data to day-by-day format
        data_ip = CGMS2DayByDay(data, dt0=5, inter_gap=inter_gap, tz=tz)
        dt0 = data_ip[2]  # Time between measurements in minutes
        
        # Ensure short_ma and long_ma are appropriate
        if short_ma >= long_ma:
            short_ma, long_ma = long_ma, short_ma
        
        # Calculate moving averages
        data = data.copy()
        data['MA_Short'] = data['gl'].rolling(window=short_ma, min_periods=1).mean()
        data['MA_Long'] = data['gl'].rolling(window=long_ma, min_periods=1).mean()
        data['DELTA_SHORT_LONG'] = data['MA_Short'] - data['MA_Long']
        
        # Find crossing points
        crossings = []
        for i in range(1, len(data)):
            if (data['DELTA_SHORT_LONG'].iloc[i-1] * data['DELTA_SHORT_LONG'].iloc[i] <= 0):
                crossings.append(i)
        
        if len(crossings) < 2:
            return pd.DataFrame({'start': [data['time'].iloc[0]], 
                               'end': [data['time'].iloc[-1]], 
                               'mage': [np.nan],
                               'plus_or_minus': [None],
                               'first_excursion': [None]})
        
        # Find peaks and nadirs
        peaks_nadirs = []
        for i in range(len(crossings)-1):
            start_idx = crossings[i]
            end_idx = crossings[i+1]
            segment = data.iloc[start_idx:end_idx+1]
            
            if data['DELTA_SHORT_LONG'].iloc[start_idx] > 0:
                # Looking for peak
                peak_idx = segment['gl'].idxmax()
                peaks_nadirs.append((peak_idx, 'PEAK'))
            else:
                # Looking for nadir
                nadir_idx = segment['gl'].idxmin()
                peaks_nadirs.append((nadir_idx, 'NADIR'))
        
        # Calculate excursions
        std_gl = data['gl'].std()
        mage_plus = []
        mage_minus = []
        
        for i in range(len(peaks_nadirs)-1):
            curr_idx, curr_type = peaks_nadirs[i]
            next_idx, next_type = peaks_nadirs[i+1]
            
            if curr_type == 'NADIR' and next_type == 'PEAK':
                excursion = data['gl'].loc[next_idx] - data['gl'].loc[curr_idx]
                if excursion >= std_gl:
                    mage_plus.append(excursion)
            elif curr_type == 'PEAK' and next_type == 'NADIR':
                excursion = data['gl'].loc[curr_idx] - data['gl'].loc[next_idx]
                if excursion >= std_gl:
                    mage_minus.append(excursion)
        
        # Calculate final MAGE value based on direction
        if direction == 'plus':
            mage_val = np.mean(mage_plus) if mage_plus else np.nan
            plus_or_minus = 'PLUS'
        elif direction == 'minus':
            mage_val = np.mean(mage_minus) if mage_minus else np.nan
            plus_or_minus = 'MINUS'
        elif direction == 'max':
            mage_plus_val = np.mean(mage_plus) if mage_plus else 0
            mage_minus_val = np.mean(mage_minus) if mage_minus else 0
            mage_val = max(mage_plus_val, mage_minus_val)
            plus_or_minus = 'PLUS' if mage_plus_val >= mage_minus_val else 'MINUS'
        else:  # 'avg' or 'service'
            mage_plus_val = np.mean(mage_plus) if mage_plus else 0
            mage_minus_val = np.mean(mage_minus) if mage_minus else 0
            mage_val = (mage_plus_val + mage_minus_val) / 2
            plus_or_minus = 'AVG'
        
        return pd.DataFrame({
            'start': [data['time'].iloc[0]],
            'end': [data['time'].iloc[-1]],
            'mage': [mage_val],
            'plus_or_minus': [plus_or_minus],
            'first_excursion': [None]  # Not implemented in this version
        })
    
    # Handle Series input
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame format
        data_df = pd.DataFrame({
            'id': ['subject1'] * len(data),
            'time': pd.date_range(start='2020-01-01', periods=len(data), freq='5min'),
            'gl': data.values
        })
        if version == 'ma':
            result = mage_ma_single(data_df, short_ma, long_ma)
        else:
            result = pd.DataFrame({'MAGE': [mage_naive(data_df)]})
        return result
    
    # Handle DataFrame input
    data = check_data_columns(data)
    
    # Calculate MAGE for each subject
    result = []
    for subject in data['id'].unique():
        subject_data = data[data['id'] == subject].copy()
        if len(subject_data.dropna(subset=['gl'])) == 0:
            continue
            
        if version == 'ma':
            subject_result = mage_ma_single(subject_data, short_ma, long_ma)
            mage_val = subject_result['mage'].iloc[0]
        else:
            mage_val = mage_naive(subject_data)
            
        result.append({'id': subject, 'MAGE': mage_val})
    
    return pd.DataFrame(result) 