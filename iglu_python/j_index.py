from typing import Union

import pandas as pd

from .utils import check_data_columns


def j_index(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate J-Index score for glucose measurements.

    The function produces a DataFrame with values equal to the J-Index score,
    which is calculated as 0.001 * (mean(G) + sd(G))^2 where G is the list of
    glucose measurements.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for J-Index value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Wojcicki (1995) "J"-index. A new proposition of the assessment
    of current glucose control in diabetic patients
    Hormone and Metabolic Research 27:41-42,
    doi:10.1055/s-2007-979906.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> j_index(data)
       id    J_index
    0  subject1  1.5625
    1  subject2  1.4400

    >>> j_index(data['gl'])
       J_index
    0  1.5000
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        # Calculate mean and standard deviation
        mean_gl = data.mean()
        sd_gl = data.std()

        # Calculate J-index
        j_index = 0.001 * (mean_gl + sd_gl) ** 2

        return pd.DataFrame({"J_index": [j_index]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Initialize result list
    result = []

    # Process each subject
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject]

        # Calculate mean and standard deviation
        mean_gl = subject_data["gl"].mean()
        sd_gl = subject_data["gl"].std()

        # Calculate J-index
        j_index = 0.001 * (mean_gl + sd_gl) ** 2

        result.append({"id": subject, "J_index": j_index})

    # Convert to DataFrame
    return pd.DataFrame(result)
