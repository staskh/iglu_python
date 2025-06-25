from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def mad_glu(data: Union[pd.DataFrame, pd.Series, np.ndarray, list], constant: float = 1.4826) -> pd.DataFrame | float:
    """
    Calculate Median Absolute Deviation (MAD) of glucose values.

    The function produces MAD values in a DataFrame. MAD is calculated by taking
    the median of the difference of the glucose readings from their median and
    multiplying it by a scaling factor.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
    constant : float, default=1.4826
        Scaling factor to multiply the MAD value. The default value of 1.4826
        makes the MAD consistent with the standard deviation for normally
        distributed data.

    Returns
    -------
    pd.DataFrame|float
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - MAD: MAD value (median absolute deviation of glucose values).
        If a Series of glucose values is passed, then a float is returned.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> mad_glu(data)
       id    MAD
    0  subject1  25.0
    1  subject2  30.0

    >>> mad_glu(data['gl'])
       MAD
    0  27.5
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return mad_glu_single(data, constant)

    # Handle DataFrame input
    data = check_data_columns(data)

    out = data.groupby("id").agg(MAD=("gl", lambda x: mad_glu_single(x, constant))).reset_index()
    return out


def mad_glu_single(gl: pd.Series, constant: float = 1.4826) -> float:
    """
    Calculate Median Absolute Deviation (MAD) of glucose values for a single subject.
    """
    gl = gl.dropna()
    if len(gl) == 0:
        return np.nan
    mad_val = np.median(np.abs(gl - np.median(gl))) * constant
    return mad_val
