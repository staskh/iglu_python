from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def range_glu(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> pd.DataFrame|float:
    """
    Calculate glucose level range.

    The function outputs the distance between minimum and maximum glucose values
    per subject in a DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values

    Returns
    -------
    pd.DataFrame|float
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - range: range of glucose values (max - min). If a Series of glucose values is passed, then a float is returned.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> range_glu(data)
       id  range
    0  subject1     50
    1  subject2     60

    >>> range_glu(data['gl'])
       range
    0     70
    """
    # Handle Series input
    if isinstance(data, (pd.Series, np.ndarray, list)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        # Calculate range for Series
        range_val = float(data.max() - data.min())
        return range_val

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate range for each subject
    result = (
        data.groupby("id").agg(range=("gl", lambda x: x.max() - x.min())).reset_index()
    )

    return result
