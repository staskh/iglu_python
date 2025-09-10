from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def grade(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate mean GRADE score for each subject.

    The function produces a DataFrame with values equal to the mean GRADE score
    for each subject. The output columns correspond to the subject id and GRADE
    value, and the output rows correspond to the subjects.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for GRADE value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Hill et al. (2007): A method for assessing quality of control
    from glucose profiles
    Diabetic Medicine 24: 753-758,
    doi:10.1111/j.1464-5491.2007.02119.x.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> grade(data)
       id     GRADE
    0  subject1  23.45
    1  subject2  21.23

    >>> grade(data['gl'])
       GRADE
    0  22.34
    """
    # Handle Series input
    if isinstance(data, (pd.Series, np.ndarray, list)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return np.mean(_grade_formula(data.dropna()))

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate GRADE score for each subject
    result = data.groupby("id")[["gl"]].apply(lambda x: np.mean(_grade_formula(x["gl"].dropna()))).reset_index()
    result.columns = ["id", "GRADE"]

    return result


def _grade_formula(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Helper function to calculate GRADE score for individual glucose values.

    Parameters
    ----------
    x : Union[pd.Series, np.ndarray]
        Glucose values in mg/dL

    Returns
    -------
    Union[pd.Series, np.ndarray]
        GRADE scores for each glucose value
    """
    grade = 425 * (np.log10(np.log10(x / 18)) + 0.16) ** 2
    return np.minimum(grade, 50)  # Cap at 50
