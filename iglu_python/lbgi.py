import pandas as pd
import numpy as np
from typing import Union
from .utils import check_data_columns

def calculate_lbgi(glucose_values: pd.Series) -> float:
    """
    Calculate LBGI for a single series of glucose values.
    
    Parameters
    ----------
    glucose_values : pd.Series
        Series of glucose values in mg/dL
        
    Returns
    -------
    float
        LBGI value
    """
    # Remove NaN values
    glucose_values = glucose_values.dropna()
    
    if len(glucose_values) == 0:
        return np.nan
    
    # Calculate LBGI using the formula from the R implementation
    # LBGI = 22.77 * mean(fbg[gl < 112.5]^2)
    # where fbg = max(0, 1.509 * (log(gl)^1.084 - 5.381))
    
    # Filter values below threshold
    low_glucose = glucose_values[glucose_values < 112.5]
    
    if len(low_glucose) == 0:
        return 0.0
    
    # Calculate fbg for low glucose values
    fbg = np.maximum(0, 1.509 * (np.log(low_glucose) ** 1.084 - 5.381))
    
    # Calculate LBGI
    lbgi = 22.77 * np.mean(fbg ** 2)
    
    return lbgi

def lbgi(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate the Low Blood Glucose Index (LBGI) for each subject.
    
    The LBGI is calculated using the formula from the R implementation:
    LBGI = 22.77 * mean(fbg[gl < 112.5]^2)
    where fbg = max(0, 1.509 * (log(gl)^1.084 - 5.381))
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns ['id', 'time', 'gl'] or Series of glucose values
        in mg/dL
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['id', 'LBGI'] containing LBGI values for each subject
        If input is a Series, returns DataFrame with single row and column 'LBGI'
        
    References
    ----------
    Kovatchev BP, Cox DJ, Gonder-Frederick LA, Young-Hyman D, Schlundt D, Clarke WL.
    Assessment of risk for severe hypoglycemia among adults with IDDM: validation of
    the low blood glucose index. Diabetes Care. 1998;21(11):1870-1875.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from iglu_python.lbgi import lbgi
    >>> 
    >>> # Example with DataFrame input
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
    ...     'gl': [80, 90, 70, 60]
    ... })
    >>> result = lbgi(data)
    >>> print(result)
           id       LBGI
    0  subject1  0.123456
    1  subject2  0.234567
    >>> 
    >>> # Example with Series input
    >>> data = pd.Series([80, 90, 70, 60])
    >>> result = lbgi(data)
    >>> print(result)
           LBGI
    0  0.123456
    """
    if isinstance(data, pd.Series):
        lbgi_value = calculate_lbgi(data)
        return pd.DataFrame({'LBGI': [lbgi_value]})
    
    # Check DataFrame format
    check_data_columns(data)
    
    if len(data) == 0:
        raise ValueError("Empty DataFrame provided")
    
    # Calculate LBGI for each subject
    result = pd.DataFrame(columns=['id', 'LBGI'])
    
    for subject_id in data['id'].unique():
        subject_data = data[data['id'] == subject_id]['gl']
        lbgi_value = calculate_lbgi(subject_data)
        result = pd.concat([result, pd.DataFrame({
            'id': [subject_id],
            'LBGI': [lbgi_value]
        })], ignore_index=True)
    
    return result 