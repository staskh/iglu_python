from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def j_index(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> pd.DataFrame | float:
    """
    Calculate J-Index score for glucose measurements.

    The function produces a DataFrame with values equal to the J-Index score,
    which is calculated as 0.001 * (mean(G) + sd(G))^2 where G is the list of
    glucose measurements.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for J-Index value. If a Series of glucose values is passed, then a float is returned.

    References
    ----------
    Wojcicki (1995) "J"-index. A new proposition of the assessment
    of current glucose control in diabetic patients
    Hormone and Metabolic Research 27:41-42,
    doi:10.1055/s-2007-979906.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> j_index(data)
       id    J_index
    0  subject1  1.5625
    1  subject2  1.4400

    >>> j_index(data['gl'])
       J_index
    0  1.5000
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return j_index_single(data)

    # Handle DataFrame input
    data = check_data_columns(data)

    out = data.groupby("id").agg(J_index=("gl", lambda x: j_index_single(x))).reset_index()
    return out


def j_index_single(gl: pd.Series) -> float:
    """
    Calculate J-Index score for a single subject.
    """
    gl = gl.dropna()
    if len(gl) == 0:
        return np.nan
    # Calculate mean and standard deviation
    mean_gl = gl.mean()
    sd_gl = gl.std()

    # Calculate J-index
    j_index = 0.001 * (mean_gl + sd_gl) ** 2
    return j_index
