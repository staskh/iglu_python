from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def hypo_index(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list], LLTR: int = 80, b: float = 2, d: int = 30
) -> pd.DataFrame | float:
    """
    Calculate Hypoglycemia Index.

    The function produces Hypoglycemia Index values in a DataFrame object. The Hypoglycemia
    Index is calculated by taking the sum of the differences between the lower limit of
    target range (LLTR) and glucose values below the LLTR, raised to power b, divided by
    the product of the number of measurements and a scaling factor d.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
    LLTR : int, default=80
        Lower Limit of Target Range, in mg/dL
    b : float, default=2
        Exponent, generally in the range from 1.0 to 2.0
    d : int, default=30
        Scaling factor, to display Hyperglycemia Index, Hypoglycemia Index, and IGC on
        approximately the same numerical range as measurements of HBGI, LBGI and GRADE

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the Hypoglycemia Index value. If a Series of glucose values is passed,
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
    ...     'gl': [70, 60, 75, 65]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> hypo_index(data)
       id  hypo_index
    0  subject1  0.123
    1  subject2  0.089

    >>> hypo_index(data['gl'])
       hypo_index
    0  0.106
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return hypo_index_single(data, LLTR, b, d)

    data = check_data_columns(data)
    out = data.groupby("id").agg(hypo_index=("gl", lambda x: hypo_index_single(x, LLTR, b, d))).reset_index()
    return out


def hypo_index_single(gl: pd.Series, LLTR: int = 80, b: float = 2, d: int = 30) -> float:
    """
    Calculate Hypoglycemia Index for a single subject.
    """
    gl = gl.dropna()
    if len(gl) == 0:
        return np.nan
    # Calculate hypo_index
    hypo_values = LLTR - gl[gl < LLTR]
    hypo_index = np.sum(hypo_values**b) / (len(gl) * d)
    return hypo_index
