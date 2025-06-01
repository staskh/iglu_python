from typing import Union

import pandas as pd

from .utils import check_data_columns


def sd_glu(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate standard deviation of glucose values.

    The function produces a DataFrame with values equal to the standard deviation
    of glucose measurements. The output columns correspond to the subject id and
    the standard deviation value, and the output rows correspond to the subjects.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the standard deviation value. If a Series of glucose values is passed,
        then a DataFrame without the subject id is returned.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> sd_glu(data)
       id         SD
    0  subject1  35.36
    1  subject2  42.43

    >>> sd_glu(data['gl'])
           SD
    0  38.89
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        return pd.DataFrame({"SD": [data.std()]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate standard deviation for each subject
    out = data.groupby("id")["gl"].std().reset_index()
    out.columns = ["id", "SD"]

    return out
