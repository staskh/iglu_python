from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def ea1c(data: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> pd.DataFrame | float:
    """
    Calculate estimated A1C (eA1C) values.

    The function produces a DataFrame with values equal to the estimated A1C
    calculated from mean glucose values. The eA1C score is calculated by
    (46.7 + mean(G))/28.7 where G is the vector of Glucose Measurements (mg/dL).

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, list, np.ndarray]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for eA1C values. If a Series of glucose values is passed, then a float value
        is returned.

    References
    ----------
    Nathan (2008) Translating the A1C assay into estimated average glucose values
    Hormone and Metabolic Research 31: 1473-1478,
    doi:10.2337/dc08-0545.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> ea1c(data)
       id       eA1C
    0  subject1  7.89
    1  subject2  7.45

    >>> ea1c(data['gl'])
       eA1C
    0  7.67
    """
    # Handle Series input
    if isinstance(data, (pd.Series, np.ndarray, list)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return ea1c_single(data)

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate eA1C for each subject
    out = data.groupby("id").agg(eA1C=("gl", lambda x: ea1c_single(x))).reset_index()

    return out


def ea1c_single(data: pd.Series) -> float:
    """Calculate eA1C for a single subject"""
    if not isinstance(data, pd.Series):
        raise ValueError("Data must be a pandas Series")

    data = data.dropna()
    if len(data) == 0:
        return np.nan

    return (46.7 + data.mean()) / 28.7
