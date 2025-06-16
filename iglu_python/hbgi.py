from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def hbgi(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> pd.DataFrame|float:
    r"""
    Calculate High Blood Glucose Index (HBGI).

    The function produces a DataFrame with values equal to the HBGI, which is calculated
    by  22.77 * sum(fbg[gl >= 112.5]^2) / n, where fbg = max(0, (log(gl)^1.084 - 5.381)),
    gl is the glucose measurement, and n is the total number of measurements.

    TODO: Review description from R implementation documentation:
    HBGI is calculated by :math:`1/n * \sum (10 * fg_i ^2)`,
    where :math:`fg_i = \max(0, 1.509 * (\log(G_i)^{1.084} - 5.381))`,
    G_i is the ith Glucose measurement for a subject, and
    n is the total number of measurements for that subject.

    Apparently, both calculations are equivalent.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for HBGI values. If a Series of glucose values is passed, then a float is returned.

    References
    ----------
    Kovatchev et al. (2006) Evaluation of a New Measure of Blood Glucose Variability in,
    Diabetes
    Diabetes care 29: 2433-2438,
    doi:10.2337/dc06-1085.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> hbgi(data)
       id       HBGI
    0  subject1  5.67
    1  subject2  4.23

    >>> hbgi(data['gl'])
       HBGI
    0  4.95
    """

    # Handle Series input
    if isinstance(data, (pd.Series, np.ndarray, list)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return calculate_hbgi_single(data)

    # Handle DataFrame input
    data = check_data_columns(data)

    out = data.groupby('id').agg(
        HBGI = ("gl", lambda x: calculate_hbgi_single(x))
    ).reset_index()
    return out

def calculate_hbgi_single(glucose_values: pd.Series) -> float:
    """Helper function to calculate HBGI for a single series of values."""
    glucose_values = glucose_values.dropna()
    if len(glucose_values) == 0:
        return np.nan

    # Calculate fbg values
    fbg = 1.509 * (np.log(glucose_values) ** 1.084 - 5.381)
    fbg = np.maximum(fbg, 0)  # Take max with 0

    # Calculate HBGI
    n = len(glucose_values)
    hbgi_value = 10 * np.sum(fbg[glucose_values >= 112.5] ** 2) / n

    return hbgi_value
