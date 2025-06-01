from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def mad_glu(
    data: Union[pd.DataFrame, pd.Series], constant: float = 1.4826
) -> pd.DataFrame:
    """
    Calculate Median Absolute Deviation (MAD) of glucose values.

    The function produces MAD values in a DataFrame. MAD is calculated by taking
    the median of the difference of the glucose readings from their median and
    multiplying it by a scaling factor.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    constant : float, default=1.4826
        Scaling factor to multiply the MAD value. The default value of 1.4826
        makes the MAD consistent with the standard deviation for normally
        distributed data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - MAD: MAD value (median absolute deviation of glucose values)

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
    if isinstance(data, pd.Series):
        # Calculate MAD for the Series
        mad_val = np.median(np.abs(data - np.median(data))) * constant
        return pd.DataFrame({"MAD": [mad_val]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate MAD for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject]
        if len(subject_data.dropna(subset=["gl"])) == 0:
            continue

        # Calculate MAD for this subject
        mad_val = (
            np.median(np.abs(subject_data["gl"] - np.median(subject_data["gl"])))
            * constant
        )
        result.append({"id": subject, "MAD": mad_val})

    return pd.DataFrame(result)
