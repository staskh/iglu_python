import pandas as pd
import numpy as np
from typing import Union
from .utils import check_data_columns

def calculate_sd_roc(glucose_values: pd.Series, timestamps: pd.Series) -> float:
    """
    Calculate SD of ROC for a single series of glucose values.
    
    Parameters
    ----------
    glucose_values : pd.Series
        Series of glucose values in mg/dL
    timestamps : pd.Series
        Series of timestamps corresponding to glucose values
        
    Returns
    -------
    float
        Standard deviation of rate of change
    """
    # Remove NaN values
    mask = ~(glucose_values.isna() | timestamps.isna())
    glucose_values = glucose_values[mask]
    timestamps = timestamps[mask]
    
    if len(glucose_values) < 2:
        return np.nan
    
    # Sort by timestamp
    sort_idx = timestamps.argsort()
    glucose_values = glucose_values.iloc[sort_idx]
    timestamps = timestamps.iloc[sort_idx]
    
    # Calculate time differences in minutes
    time_diffs = np.diff(timestamps.astype(np.int64) // 10**9) / 60.0
    
    # Calculate glucose differences
    glucose_diffs = np.diff(glucose_values)
    
    # Calculate rate of change (mg/dL per minute)
    roc = glucose_diffs / time_diffs
    
    # Calculate standard deviation of ROC
    sd_roc = np.std(roc)
    
    return sd_roc

def sd_roc(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate the Standard Deviation of Rate of Change (SD of ROC) for each subject.
    
    The SD of ROC is calculated as the standard deviation of the rate of change
    of glucose values over time. The rate of change is calculated as the difference
    in glucose values divided by the time difference between measurements.

    When calculating rate of change, missing values will be linearly interpolated
    when close enough to non-missing values.

    Calculated by taking the standard deviation of all the ROC values for each
    individual subject. NA rate of change values are omitted from the
    standard deviation calculation.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns ['id', 'time', 'gl'] or Series of glucose values
        in mg/dL. If Series is provided, it must have a datetime index.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['id', 'sd_roc'] containing SD of ROC values for each subject
        If input is a Series, returns DataFrame with single row and column 'SD_ROC'
        
    References
    ----------
    Kovatchev BP, Cox DJ, Gonder-Frederick LA, Young-Hyman D, Schlundt D, Clarke WL.
    Assessment of risk for severe hypoglycemia among adults with IDDM: validation of
    the low blood glucose index. Diabetes Care. 1998;21(11):1870-1875.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from iglu_python.sd_roc import sd_roc
    >>> 
    >>> # Example with DataFrame input
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
    ...     'gl': [100, 120, 100, 80]
    ... })
    >>> result = sd_roc(data)
    >>> print(result)
           id    SD_ROC
    0  subject1  4.000000
    1  subject2  4.000000
    >>> 
    >>> # Example with Series input
    >>> data = pd.Series([100, 120, 100, 80],
    ...                  index=pd.to_datetime(['2020-01-01 00:00:00',
    ...                                       '2020-01-01 00:05:00',
    ...                                       '2020-01-01 00:10:00',
    ...                                       '2020-01-01 00:15:00']))
    >>> result = sd_roc(data)
    >>> print(result)
           SD_ROC
    0  4.000000
    """
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series input must have a datetime index")
        sd_roc_value = calculate_sd_roc(data, pd.Series(data.index))
        return pd.DataFrame({'SD_ROC': [sd_roc_value]})
    
    # Check DataFrame format
    check_data_columns(data)
    
    if len(data) == 0:
        raise ValueError("Empty DataFrame provided")
    
    # Calculate SD of ROC for each subject
    result = pd.DataFrame(columns=['id', 'sd_roc'])
    
    for subject_id in data['id'].unique():
        subject_data = data[data['id'] == subject_id]
        sd_roc_value = calculate_sd_roc(subject_data['gl'], subject_data['time'])
        result = pd.concat([result, pd.DataFrame({
            'id': [subject_id],
            'sd_roc': [sd_roc_value]
        })], ignore_index=True)
    
    return result 