from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def iqr_glu(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> pd.DataFrame|float:
    """
    Calculate glucose level interquartile range (IQR).

    The function outputs the distance between the 25th percentile and 75th percentile
    of the glucose values per subject in a DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the IQR value. If a Series of glucose values is passed, then a float is returned.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> iqr_glu(data)
       id    IQR
    0  subject1   50.0
    1  subject2   60.0

    >>> iqr_glu(data['gl'])
       IQR
    0   70.0
    """
    # Handle Series input
    if isinstance(data, (pd.Series,list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        data = data.dropna()
        if len(data) == 0:
            return np.nan
        # Calculate IQR for Series
        iqr_val = iqr_glu_single(data)
        return iqr_val

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate IQR for each subject
    # drop all rows with missing values
    data = data.dropna()
    result = (
        data.groupby("id")
        .agg(IQR=("gl", lambda x: iqr_glu_single(x)))
        .reset_index()
    )

    return result

def iqr_glu_single(
    gl: pd.Series,
) -> float:
    """
    Calculate glucose level interquartile range (IQR) for a single subject.

    Parameters
    ----------
    gl : pd.Series
        Series of glucose values

    Returns
    """
    gl = gl.dropna()
    if len(gl) == 0:
        return np.nan
    # Calculate IQR for Series
    iqr_val = np.percentile(gl, 75) - np.percentile(gl, 25)
    return iqr_val
