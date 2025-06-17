
import numpy as np
import pandas as pd

from .utils import check_data_columns


def adrr(data: pd.DataFrame|pd.Series) -> pd.DataFrame|float:
    """
    Calculate average daily risk range (ADRR)

    The function `adrr` produces ADRR values in a DataFrame object.

    Parameters
    ----------
    data : pd.DataFrame|pd.Series
        DataFrame object with column names "id", "time", and "gl".
        or a Timeseries of glucose values.

    Returns
    -------
    pd.DataFrame|float
        A DataFrame object with two columns: subject id and corresponding
        ADRR value. or a float value for a Timeseries of glucose values.

    Details
    -------
    A DataFrame object with 1 row for each subject, a column for subject id and
    a column for ADRR values is returned. `NaN` glucose values are
    omitted from the calculation of the ADRR values.

    ADRR is the average sum of HBGI corresponding to the highest glucose
    value and LBGI corresponding to the lowest glucose value for each day,
    with the average taken over the daily sums. If there are no high glucose or
    no low glucose values, then 0 will be substituted for the HBGI value or the
    LBGI value, respectively, for that day.

    References
    ----------
    Kovatchev et al. (2006) Evaluation of a New Measure of Blood Glucose Variability in,
    Diabetes. Diabetes care 29:2433-2438.
    DOI: 10.2337/dc06-1085

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> import iglu_python as iglu
    >>>
    >>> # Example data
    >>> data = pd.read_csv('tests/data/example_data_1_subject.csv',index_col=0)
    >>> iglu.adrr(data)
    """


    # Validate input
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        return adrr_single(data)

    data = check_data_columns(data)

    data.set_index("time", inplace=True,drop=True)
    out = data.groupby("id").agg(
        ADRR = ("gl", lambda x: adrr_single(x))
    ).reset_index()

    return out


def adrr_single(data: pd.DataFrame|pd.Series) -> float:
    """Internal function to calculate ADRR for a single subject or timeseries of glucose values"""

    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
    elif isinstance(data, pd.DataFrame):
        data = data.set_index("time")["gl"]
    else:
        raise ValueError("Data  must be a pandas DataFrame or Series")

    data_filtered = data.dropna()
    if len(data_filtered) == 0:
        return np.nan

    # Group by date and calculate daily risk for each day
    daily_risks = data_filtered.groupby(data_filtered.index.date).apply(
        lambda x: _calculate_daily_risk(x)
    )
    return daily_risks.mean()

def _calculate_daily_risk(gl: pd.Series) -> float:
    """Calculate daily risk range for a single day and subject"""

    # Calculate BGI (Blood Glucose Index)
    bgi = (np.log(gl) ** 1.084) - 5.381

    # Calculate max and min BGI values for the day
    max_bgi = np.maximum(bgi.max(), 0)
    min_bgi = np.minimum(bgi.min(), 0)

    # Calculate risk components
    max_risk = 22.77 * (max_bgi**2)
    min_risk = 22.77 * (min_bgi**2)

    # Daily risk range is the sum of max and min risks
    drr = min_risk + max_risk

    return drr
