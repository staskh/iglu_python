from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def iqr_glu(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate glucose level interquartile range (IQR).

    The function outputs the distance between the 25th percentile and 75th percentile
    of the glucose values per subject in a DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - IQR: interquartile range of glucose values (75th percentile - 25th percentile)

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
    if isinstance(data, pd.Series):
        # Calculate IQR for Series
        iqr_val = np.percentile(data, 75) - np.percentile(data, 25)
        return pd.DataFrame({"IQR": [iqr_val]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate IQR for each subject
    # drop all rows with missing values
    data = data.dropna()
    result = (
        data.groupby("id")
        .agg(IQR=("gl", lambda x: np.percentile(x, 75) - np.percentile(x, 25)))
        .reset_index()
    )

    return result
