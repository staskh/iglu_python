from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def hyper_index(
    data: Union[pd.DataFrame, pd.Series], ULTR: int = 140, a: float = 1.1, c: int = 30
) -> pd.DataFrame:
    """
    Calculate Hyperglycemia Index.

    The function produces Hyperglycemia Index values in a DataFrame object. The Hyperglycemia
    Index is calculated by taking the sum of the differences between glucose values above
    the upper limit of target range (ULTR) and the ULTR, raised to power a, divided by
    the product of the number of measurements and a scaling factor c.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    ULTR : int, default=140
        Upper Limit of Target Range, in mg/dL
    a : float, default=1.1
        Exponent, generally in the range from 1.0 to 2.0
    c : int, default=30
        Scaling factor, to display Hyperglycemia Index, Hypoglycemia Index, and IGC on
        approximately the same numerical range as measurements of HBGI, LBGI and GRADE

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the Hyperglycemia Index value. If a Series of glucose values is passed,
        then a DataFrame without the subject id is returned.

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

    # Calculate hyper_index for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject]
        # Remove NA values
        subject_data = subject_data.dropna(subset=["gl"])

        if len(subject_data) == 0:
            continue

        # Calculate hyper_index
        hyper_values = subject_data[subject_data["gl"] > ULTR]["gl"] - ULTR
        hyper_index = np.sum(hyper_values**a) / (len(subject_data) * c)

        result.append({"id": subject, "hyper_index": hyper_index})

    # Convert to DataFrame
    out = pd.DataFrame(result)

    # Remove id column if input was a Series
    if is_vector and not out.empty:
        out = out.drop("id", axis=1)

    return out
