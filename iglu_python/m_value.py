from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def m_value(data: Union[pd.DataFrame, pd.Series, np.ndarray, list], r: float = 90) -> pd.DataFrame|float:
    r"""
    Calculate the M-value of Schlichtkrull et al. (1965) for each subject.

    The M-value is the mean of the logarithmic transformation of the deviation
    from a reference value. Produces a DataFrame with subject id and M-values.

    A reference value corresponding to basal glycemia in normal
    subjects; default is 90 mg/dL.

    M-value is calculated by :math:`1000 * \text{mean}(|\log_{10}(\text{gl} / r)|^3)`,
    where gl is glucose values, r is a reference value.


    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
    r : float, default=90
        A reference value corresponding to basal glycemia in normal subjects

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for M-value. If a Series of glucose values is passed, then a float is returned.

    References
    ----------
    Schlichtkrull J, Munck O, Jersild M. (1965) The M-value, an index of
    blood-sugar control in diabetics.
    Acta Medica Scandinavica 177:95-102.
    doi:10.1111/j.0954-6820.1965.tb01810.x.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> m_value(data)
       id     M_value
    0  subject1  123.45
    1  subject2   98.76

    >>> m_value(data['gl'], r=100)
       M_value
    0   111.11
    """
    # Handle Series input
    if isinstance(data, (pd.Series,list, np.ndarray)):
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        return m_value_single(data, r)

    # Handle DataFrame input
    data = check_data_columns(data)

    out = data.groupby('id').agg(
        M_value = ("gl", lambda x: m_value_single(x, r))
    ).reset_index()
    return out

def m_value_single(gl:  pd.Series, r: float = 90) -> float:
    """
    Calculate the M-value of Schlichtkrull et al. (1965) for a single subject.
    """
    gl = gl.dropna()
    if len(gl) == 0:
        return np.nan
    m_value = 1000 * np.mean(np.abs(np.log10(gl / r)) ** 3)
    return m_value

