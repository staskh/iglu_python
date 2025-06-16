from typing import Union

import numpy as np
import pandas as pd

from .hyper_index import hyper_index
from .hypo_index import hypo_index
from .utils import check_data_columns


def igc(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    LLTR: int = 80,
    ULTR: int = 140,
    a: float = 1.1,
    b: float = 2,
    c: int = 30,
    d: int = 30,
) -> pd.DataFrame|float:
    """
    Calculate Index of Glycemic Control (IGC).

    The function produces IGC values in a DataFrame object. IGC is calculated by taking
    the sum of the Hyperglycemia Index and the Hypoglycemia Index.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
    LLTR : int, default=80
        Lower Limit of Target Range, in mg/dL
    ULTR : int, default=140
        Upper Limit of Target Range, in mg/dL
    a : float, default=1.1
        Exponent for hyperglycemia calculation, generally in the range from 1.0 to 2.0
    b : float, default=2
        Exponent for hypoglycemia calculation, generally in the range from 1.0 to 2.0
    c : int, default=30
        Scaling factor for hyperglycemia index
    d : int, default=30
        Scaling factor for hypoglycemia index

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the IGC value. If a Series of glucose values is passed, then a float is returned.

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
    >>> igc(data)
       id        IGC
    0  subject1  0.123
    1  subject2  0.089

    >>> igc(data['gl'])
       IGC
    0  0.106
    """
    # Handle Series input
    if isinstance(data, (pd.Series,list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return igc_single(data, LLTR, ULTR, a, b, c, d)

    # Check and prepare data
    data = check_data_columns(data)

    out = data.groupby('id').agg(
        IGC = ("gl", lambda x: igc_single(x, LLTR, ULTR, a, b, c, d))
    ).reset_index()
    return out

def igc_single(
    gl: pd.Series,
    LLTR: int = 80,
    ULTR: int = 140,
    a: float = 1.1,
    b: float = 2,
    c: int = 30,
    d: int = 30
) -> float:
    """
    Calculate Index of Glycemic Control for a single subject.
    """
        # Calculate hyper_index and hypo_index
    out_hyper = hyper_index(gl, ULTR=ULTR, a=a, c=c)
    out_hypo = hypo_index(gl, LLTR=LLTR, b=b, d=d)

    out = out_hyper + out_hypo
    return out
