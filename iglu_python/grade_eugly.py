from typing import Union

import numpy as np
import pandas as pd

from .grade import _grade_formula
from .utils import check_data_columns


def grade_eugly(
    data: Union[pd.DataFrame, pd.Series], lower: int = 70, upper: int = 140
) -> pd.DataFrame:
    """
    Calculate percentage of GRADE score attributable to target range.

    The function produces a DataFrame with values equal to the percentage of GRADE score
    attributed to values in the target range, i.e. values not below hypoglycemic or
    above hyperglycemic cutoffs.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    lower : int, default=70
        Lower bound used for hypoglycemia cutoff, in mg/dL
    upper : int, default=140
        Upper bound used for hyperglycemia cutoff, in mg/dL

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for GRADE euglycemia value. If a Series of glucose values is passed, then a DataFrame
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
    >>> grade_eugly(data)
       id  GRADE_eugly
    0  subject1      45.67
    1  subject2      78.90

    >>> grade_eugly(data['gl'], lower=80, upper=180)
       GRADE_eugly
    0       65.43
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame({"GRADE_eugly": [np.nan]})

        # Calculate GRADE scores
        grade_scores = _grade_formula(data)

        # Calculate percentage in target range
        in_range = (data >= lower) & (data <= upper)
        total_grade = np.sum(grade_scores)
        if total_grade == 0:
            return pd.DataFrame({"GRADE_eugly": [np.nan]})

        eugly_percent = (np.sum(grade_scores[in_range]) / total_grade) * 100
        return pd.DataFrame({"GRADE_eugly": [eugly_percent]})

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate GRADE euglycemia for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].dropna(subset=["gl"])
        if len(subject_data) == 0:
            continue

        # Calculate GRADE scores
        grade_scores = _grade_formula(subject_data["gl"])

        # Calculate percentage in target range
        in_range = (subject_data["gl"] >= lower) & (subject_data["gl"] <= upper)
        total_grade = np.sum(grade_scores)
        if total_grade == 0:
            continue

        eugly_percent = (np.sum(grade_scores[in_range]) / total_grade) * 100
        result.append({"id": subject, "GRADE_eugly": eugly_percent})

    return pd.DataFrame(result)
