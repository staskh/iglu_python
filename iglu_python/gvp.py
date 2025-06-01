from typing import Union

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns


def calculate_gvp(glucose_values: pd.Series, timestamps: pd.Series) -> float:
    """
    Calculate GVP for a single series of glucose values.

    Parameters
    ----------
    glucose_values : pd.Series
        Series of glucose values in mg/dL
    timestamps : pd.Series
        Series of timestamps corresponding to glucose values

    Returns
    -------
    float
        Glucose Variability Percentage
    """
    # Remove NaN values
    mask = ~(glucose_values.isna() | timestamps.isna())
    glucose_values = glucose_values[mask]
    timestamps = timestamps[mask]

    if len(glucose_values) < 2:
        return np.nan

    # Sort by timestamp
    sort_idx = timestamps.argsort()
    glucose_values = glucose_values.iloc[sort_idx]
    timestamps = timestamps.iloc[sort_idx]

    # Calculate time differences in minutes
    time_diffs = np.diff(timestamps.astype(np.int64) // 10**9) / 60.0

    # Calculate glucose differences
    glucose_diffs = np.diff(glucose_values)

    # Calculate total length of glucose trace
    added_length = np.sqrt(time_diffs**2 + glucose_diffs**2)
    total_length = np.sum(added_length)

    # Calculate length of flat trace
    base_length = np.sum(time_diffs)

    # Calculate GVP
    gvp = (total_length / base_length - 1) * 100

    return gvp


def gvp(data: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> pd.DataFrame:
    r"""
    Calculate Glucose Variability Percentage (GVP).

    The function produces a DataFrame with GVP values for each subject.

    GVP is calculated by dividing the total length of the line of the glucose trace
    by the length of a perfectly flat trace. The formula for this is
    :math:`\sqrt{\text{diff}^2+\text{dt0}^2}/(n*\text{dt0})`, where diff is the change in
    Glucose measurements from one reading to the next. NA glucose values are
    omitted from the calculation of the GVP.
    dt0 is the time gap between measurements and n is the number of glucose readings


    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for GVP value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Peyser et al. (2017) Glycemic Variability Percentage: A Novel Method for Assessing
    Glycemic Variability from Continuous Glucose Monitor Data.
    Diabetes Technol Ther 20(1):6–16,
    doi:10.1089/dia.2017.0187.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> gvp(data)
       id       GVP
    0  subject1  45.67
    1  subject2  38.92

    >>> gvp(data['gl'])
       GVP
    0  42.30
    """
    # Handle Series input
    is_vector = False
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    if isinstance(data, pd.Series):
        is_vector = True
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame({"GVP": [np.nan]})

        # Convert to DataFrame format for processing
        data = pd.DataFrame(
            {
                "id": ["subject1"] * len(data),
                "time": pd.date_range(
                    start="2020-01-01", periods=len(data), freq="5min"
                ),
                "gl": data.values,
            }
        )

    # Handle DataFrame input
    data = check_data_columns(data)

    def gvp_single(subj_data):
        """Calculate GVP for a single subject"""
        # Get interpolated data
        daybyday, _, reading_gap = CGMS2DayByDay(subj_data)
        daybyday = daybyday.flatten()

        # Calculate differences between consecutive readings
        diffvec = np.diff(daybyday)
        # Exclude NA values from diffvec
        diffvec = diffvec[~np.isnan(diffvec)]

        # Calculate added length (hypotenuse) and base length
        added_length = np.sqrt(reading_gap**2 + diffvec**2)
        base_length = len(diffvec) * reading_gap

        # Calculate GVP
        if base_length == 0:
            return np.nan

        return (np.sum(added_length) / base_length - 1) * 100

    # Calculate GVP for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].dropna(subset=["gl"])
        if len(subject_data) == 0:
            continue

        gvp_value = gvp_single(subject_data)
        result.append({"id": subject, "GVP": gvp_value})

    df = pd.DataFrame(result)
    if is_vector:
        df = df.drop(columns=["id"])
    return df
