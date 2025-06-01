from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def ea1c(data: Union[pd.DataFrame, pd.Series, list]) -> pd.DataFrame:
    """
    Calculate estimated A1C (eA1C) values.

    The function produces a DataFrame with values equal to the estimated A1C
    calculated from mean glucose values. The eA1C score is calculated by
    (46.7 + mean(G))/28.7 where G is the vector of Glucose Measurements (mg/dL).

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for eA1C values. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

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
    if isinstance(data, pd.Series):
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame({"eA1C": [np.nan]})

        mean_glucose = data.mean()
        ea1c_value = (46.7 + mean_glucose) / 28.7
        return pd.DataFrame({"eA1C": [ea1c_value]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate eA1C for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].dropna(subset=["gl"])
        if len(subject_data) == 0:
            continue

        mean_glucose = subject_data["gl"].mean()
        ea1c_value = (46.7 + mean_glucose) / 28.7
        result.append({"id": subject, "eA1C": ea1c_value})

    return pd.DataFrame(result)
