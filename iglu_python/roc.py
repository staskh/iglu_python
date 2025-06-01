import pandas as pd
import numpy as np
from typing import Union
from .utils import check_data_columns, CGMS2DayByDay


def roc(
    data: Union[pd.DataFrame, pd.Series],
    timelag: int = 15,
    dt0: int = 5,
    inter_gap: int = 45,
    tz: str = "",
) -> pd.DataFrame:
    """
    Calculate the Rate of Change at each time point (ROC).

    The function produces a DataFrame with values equal to the rate of change (ROC) metric.
    The output columns are subject id and ROC values. The output rows correspond to time points
    for each subject.

    The glucose values are linearly interpolated over a time grid starting at the
    beginning of the first day of data and ending on the last day of data. Because
    of this, there may be many NAs at the beginning and the end of the roc values
    for each subject. These NAs are a result of interpolated time points that do
    not have recorded glucose values near them because recording had either not
    yet begun for the day or had already ended.

    The ROC is calculated as \eqn{\frac{G(t_i) - G(t_{i-1})}{t_i - t_{i-1}}}
    where \eqn{G_i} is the Glucose measurement at time \eqn{t_i} and \eqn{G_{i-1}} is the
    Glucose measurement at time \eqn{t_{i-1}}. The time difference between the points,
    \eqn{t_i - t_{i-1}}, is selectable and set at a default of 15 minutes.

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
    pd.DataFrame
        DataFrame with a column for subject id and a column for ROC values. A ROC value is
        returned for each time point for all the subjects. If the rate of change cannot be
        calculated, the function will return NaN for that point.

    References
    ----------
    Clarke et al. (2009) Statistical Tools to Analyze Continuous Glucose Monitor Data,
    Diabetes
    Diabetes Technology and Therapeutics 11 S45-S54,
    doi:10.1089/dia.2008.0138.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> roc(data)
       id    ROC
    0  subject1  10.0
    1  subject1   NaN
    2  subject2  12.0
    3  subject2   NaN

    >>> roc(data['gl'])
       ROC
    0  10.0
    1   NaN
    2  12.0
    3   NaN
    """

    def roc_single(data: pd.DataFrame, timelag: int, dt0: int = None) -> np.ndarray:
        """Calculate ROC for a single subject's data"""
        data_ip = CGMS2DayByDay(data, dt0=dt0, inter_gap=inter_gap, tz=tz)
        gl_ip_vec = data_ip[0].flatten()  # Flatten the interpolated glucose matrix
        if dt0 is None:
            dt0 = data_ip[2]  # Get the time frequency

        if timelag < dt0:
            print(
                f"Parameter timelag cannot be less than the data collection frequency: {dt0}, "
                f"function will be evaluated with timelag = {dt0}"
            )
            timelag = dt0

        # Calculate ROC: (G(t_i) - G(t_{i-1}))/(t_i - t_{i-1})
        # First pad with NaN for the first timelag/dt0 points
        out = np.concatenate(
            [
                np.full(timelag // dt0, np.nan),
                np.diff(gl_ip_vec, n=timelag // dt0) / timelag,
            ]
        )
        return out

    # Handle Series input
    if isinstance(data, pd.Series):
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame({"ROC": [np.nan]})

        # Convert Series to DataFrame format
        data = pd.DataFrame(
            {
                "id": ["subject1"] * len(data),
                "time": pd.date_range(
                    start="2020-01-01", periods=len(data), freq=f"{dt0}T"
                ),
                "gl": data.values,
            }
        )

    # Handle DataFrame input
    data = check_data_columns(data, tz=tz)

    # Calculate ROC for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].dropna(subset=["gl"])
        if len(subject_data) == 0:
            continue

        roc_values = roc_single(subject_data, timelag)

        # Create time points for ROC values
        time_points = pd.date_range(
            start=subject_data["time"].min(), periods=len(roc_values), freq=f"{dt0}T"
        )

        # Add ROC values to result
        for t, r in zip(time_points, roc_values):
            result.append({"id": subject, "time": t, "roc": r})

    return pd.DataFrame(result)
