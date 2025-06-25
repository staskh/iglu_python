from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def hyper_index(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list], ULTR: int = 140, a: float = 1.1, c: int = 30
) -> pd.DataFrame | float:
    """
    Calculate Hyperglycemia Index.

    The function produces Hyperglycemia Index values in a DataFrame object. The Hyperglycemia
    Index is calculated by taking the sum of the differences between glucose values above
    the upper limit of target range (ULTR) and the ULTR, raised to power a, divided by
    the product of the number of measurements and a scaling factor c.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
    ULTR : int, default=140
        Upper Limit of Target Range, in mg/dL
    a : float, default=1.1
        Exponent, generally in the range from 1.0 to 2.0
    c : int, default=30
        Scaling factor, to display Hyperglycemia Index, Hypoglycemia Index, and IGC on
        approximately the same numerical range as measurements of HBGI, LBGI and GRADE

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the Hyperglycemia Index value. If a Series of glucose values is passed,
        then a float is returned.

    References
    ----------
    Rodbard (2009) Interpretation of continuous glucose monitoring data:
    glycemic variability and quality of glycemic control,
    Diabetes Technology and Therapeutics 11:55-67,
    doi:10.1089/dia.2008.0132.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> hyper_index(data)
       id  hyper_index
    0  subject1  0.123
    1  subject2  0.089

    >>> hyper_index(data['gl'])
       hyper_index
    0  0.106
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return hyper_index_single(data, ULTR, a, c)

    # Check and prepare data
    data = check_data_columns(data)

    # Calculate hyper_index for each subject
    out = data.groupby("id").agg(hyper_index=("gl", lambda x: hyper_index_single(x, ULTR, a, c))).reset_index()

    return out


def hyper_index_single(gl: pd.Series, ULTR: int = 140, a: float = 1.1, c: int = 30) -> float:
    """
    Calculate Hyperglycemia Index for a single subject.
    """
    gl = gl.dropna()
    if len(gl) == 0:
        return np.nan
    # Calculate hyper_index
    hyper_values = gl[gl > ULTR] - ULTR
    hyper_index = np.sum(hyper_values**a) / (len(gl) * c)

    return hyper_index
