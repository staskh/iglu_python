from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def median_glu(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> pd.DataFrame|float:
    """
    Calculate median glucose value for each subject.

    The function produces a DataFrame with values equal to the median glucose
    measurements for each subject. The output columns correspond to the subject id
    and median glucose value, and the output rows correspond to the subjects.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for median glucose value. If a Series of glucose values is passed, then a float is returned.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> median_glu(data)
       id  median
    0  subject1   175.0
    1  subject2   160.0

    >>> median_glu(data['gl'])
       median
    0   160.0
    """
    # Handle Series input
    if isinstance(data, (pd.Series,list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return data.median()

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate median glucose for each subject
    result = data.groupby("id")["gl"].median().reset_index()
    result.columns = ["id", "median"]

    return result
