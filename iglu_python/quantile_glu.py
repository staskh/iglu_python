from typing import List, Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def quantile_glu(
    data: Union[pd.DataFrame, pd.Series], quantiles: List[float] = [0, 25, 50, 75, 100]
) -> pd.DataFrame:
    """
    Calculate glucose level quantiles.

    The function is a wrapper for numpy's quantile function. Output is a DataFrame
    with columns for subject id and each of the quantiles. NA glucose values are
    omitted from the calculation of the quantiles.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    quantiles : List[float], default=[0, 25, 50, 75, 100]
        List of quantile values between 0 and 100

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for each quantile. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    Notes
    -----
    NA glucose values are omitted from the calculation of the quantiles.
    The values are scaled from 0-1 to 0-100 to be consistent in output with
    above_percent, below_percent, and in_range_percent.
    To scale values back to 0-1, divide the output by 100.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> quantile_glu(data)
       id     0.0    25.0    50.0    75.0   100.0
    0  subject1  150.0  162.5  175.0  187.5  200.0
    1  subject2  130.0  140.0  160.0  175.0  190.0

    >>> quantile_glu(data['gl'], quantiles=[0, 33, 66, 100])
         0.0    33.0    66.0   100.0
    0  130.0  145.0  182.5  200.0
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        # Calculate quantiles for Series
        quantile_vals = np.quantile(data.dropna(), np.array(quantiles) / 100)
        return pd.DataFrame([quantile_vals], columns=quantiles)

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate quantiles for each subject and unstack to columns
    result = (
        data.groupby("id")["gl"]
        .apply(
            lambda x: pd.Series(
                np.quantile(x.dropna(), np.array(quantiles) / 100), index=quantiles
            )
        )
        .unstack()
        .reset_index()
    )
    # Convert quantile column names to strings
    result = result.rename(columns={q: str(q) for q in quantiles})
    return result
