from typing import Union

import numpy as np
import pandas as pd

from .roc import roc
from .utils import check_data_columns


def sd_roc(
    data: Union[pd.DataFrame, pd.Series],
    timelag: int = 15,
    dt0: int = 5,
    inter_gap: int = 45,
    tz: str = "",
) -> pd.DataFrame | float:
    """
    Calculate the standard deviation of the rate of change.

    The function produces a DataFrame with the standard deviation of the rate of change
    values for each subject.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    timelag : int, default=15
        Integer indicating the time period (# minutes) over which rate of change is calculated.
        Default is 15, e.g. rate of change is the change in glucose over the past 15 minutes
        divided by 15.
    dt0 : int, optional
        The time frequency for interpolation in minutes, the default will match the data collection
        frequency.
    inter_gap : int, default=45
        Maximum gap in minutes for interpolation. The values will not be interpolated between
        the glucose measurements that are more than inter_gap minutes apart.
    tz : str, default=""
        A string specifying the time zone to be used. Empty string means current time zone.

    Returns
    -------
    pd.DataFrame|float
        DataFrame with two columns: subject id and standard deviation of the rate of change
        values for each subject.
        If a Series of glucose values is passed, then a float is returned.

    Notes
    -----
    A DataFrame with one row for each subject, a column for subject id
    and a column for the standard deviation of the rate of change.

    When calculating rate of change, missing values will be linearly interpolated
    when close enough to non-missing values.

    Calculated by taking the standard deviation of all the ROC values for each
    individual subject. NA rate of change values are omitted from the
    standard deviation calculation.

    References
    ----------
    Clarke et al. (2009) Statistical Tools to Analyze Continuous Glucose Monitor Data,
    Diabetes
    Diabetes Technology and Therapeutics 11 S45-S54,
    doi:10.1089/dia.2008.0138.

    Examples
    --------
    >>> import pandas as pd
    >>> from iglu_python import sd_roc
    >>>
    >>> # Example with DataFrame input
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject1', 'subject1'],
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:15:00',
    ...                            '2020-01-01 00:30:00', '2020-01-01 00:45:00']),
    ...     'gl': [100, 120, 100, 80]
    ... })
    >>> result = sd_roc(data)
    >>> print(result)
           id    sd_roc
    0  subject1  1.333333
    >>>
    >>> # Example with Series input
    >>> data = pd.Series([100, 120, 100, 80])
    >>> result = sd_roc(data)
    >>> print(result)
           sd_roc
    0  1.333333
    """
    # Handle Series input - convert to DataFrame format for processing
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame format
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series input must have a datetime index")
        return sd_roc_single(data, timelag, dt0, inter_gap, tz)

    # Validate input data
    data = check_data_columns(data, tz=tz)
    data.set_index("time", drop=True, inplace=True)

    # Calculate ROC values for all subjects
    results = []
    for subject_id, group in data.groupby("id"):
        sd_roc_value = sd_roc_single(group["gl"], timelag, dt0, inter_gap, tz)
        results.append({"id": subject_id, "sd_roc": sd_roc_value})

    out = pd.DataFrame(results)
    return out


def sd_roc_single(data: pd.Series, timelag: int = 15, dt0: int = 5, inter_gap: int = 45, tz: str = "") -> float:
    roc_data = roc(data, timelag=timelag, dt0=dt0, inter_gap=inter_gap, tz=tz)
    sd_roc = np.nanstd(roc_data.dropna()["roc"], ddof=1)
    return sd_roc
