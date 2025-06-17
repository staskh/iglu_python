from typing import Union

import numpy as np
import pandas as pd

from .grade import _grade_formula
from .utils import check_data_columns


def grade_hypo(data: Union[pd.DataFrame, pd.Series, np.ndarray, list], lower: int = 80) -> pd.DataFrame|float:
    """
    Calculate percentage of GRADE score attributable to hypoglycemia.

    The function produces a DataFrame with values equal to the percentage of GRADE score
    attributed to hypoglycemic glucose values, i.e. values below the specified lower bound.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
    lower : int, default=80
        Lower bound used for hypoglycemia cutoff, in mg/dL

    Returns
    -------
        pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for GRADE hypoglycemia value. If a Series of glucose values is passed, then a float
        value is returned.

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
    >>> grade_hypo(data)
       id  GRADE_hypo
    0  subject1      25.45
    1  subject2      15.67

    >>> grade_hypo(data['gl'], lower=70)
       GRADE_hypo
    0       35.43
    """
    # Handle Series input
    if isinstance(data, (pd.Series, np.ndarray, list)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return grade_hypo_single(data, lower)

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate GRADE hypoglycemia for each subject
    out = data.groupby('id').agg(
        GRADE_hypo = ("gl", lambda x: grade_hypo_single(x, lower))
    ).reset_index()

    return out

def grade_hypo_single(data: pd.Series, lower: int = 80) -> float:
    """Calculate GRADE hypoglycemia for a single timeseries"""
    data = data.dropna()
    if len(data) == 0:
        return np.nan

    # Calculate GRADE scores
    grade_scores = _grade_formula(data)

    # Calculate percentage below lower bound
    below_lower = data < lower
    total_grade = np.sum(grade_scores)
    if total_grade == 0:
        return np.nan

    hypo_percent = (np.sum(grade_scores[below_lower]) / total_grade) * 100
    return hypo_percent
