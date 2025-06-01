from typing import Union

import numpy as np
import pandas as pd

from .hyper_index import hyper_index
from .hypo_index import hypo_index
from .utils import check_data_columns


def igc(
    data: Union[pd.DataFrame, pd.Series],
    LLTR: int = 80,
    ULTR: int = 140,
    a: float = 1.1,
    b: float = 2,
    c: int = 30,
    d: int = 30,
) -> pd.DataFrame:
    """
    Calculate Index of Glycemic Control (IGC).

    The function produces IGC values in a DataFrame object. IGC is calculated by taking
    the sum of the Hyperglycemia Index and the Hypoglycemia Index.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
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
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the IGC value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

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
    is_vector = False
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    if isinstance(data, pd.Series):
        is_vector = True
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame({"GVP": [np.nan]})

        # Convert to DataFrame format for processing
        data = pd.DataFrame(
            {
                "id": ["subject1"] * len(data),
                "time": pd.date_range(
                    start="2020-01-01", periods=len(data), freq="5min"
                ),
                "gl": data.values,
            }
        )

    # Check and prepare data
    data = check_data_columns(data)

    # Calculate hyper_index and hypo_index
    out_hyper = hyper_index(data, ULTR=ULTR, a=a, c=c)
    out_hypo = hypo_index(data, LLTR=LLTR, b=b, d=d)

    # Combine the indices
    out = pd.merge(out_hyper, out_hypo, on="id")
    out["IGC"] = out["hyper_index"] + out["hypo_index"]
    out = out[["id", "IGC"]]

    # Remove id column if input was a Series
    if is_vector:
        out = out.drop("id", axis=1)

    return out
