from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def mean_glu(data: Union[pd.DataFrame, list, np.ndarray, pd.Series]) -> pd.DataFrame|float:
    """
    Calculate mean glucose value for each subject.

    The function produces a DataFrame with values equal to the mean glucose
    measurements for each subject. The output columns correspond to the subject id
    and mean glucose value, and the output rows correspond to the subjects.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for mean glucose value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> mean_glu(data)
       id  mean_glu
    0  subject1     175.0
    1  subject2     160.0

    >>> mean_glu(data['gl'])
       mean_glu
    0     157.5
    """
    # Handle Series input
    if isinstance(data, (list, np.ndarray, pd.Series)):
        if isinstance(data, (list,np.ndarray)):
            data = pd.Series(data)
        return data.mean()

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate mean glucose for each subject
    result = data.groupby("id")["gl"].mean().reset_index()
    result.columns = ["id", "mean"]

    return result
