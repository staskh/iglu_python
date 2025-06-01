from typing import List, Union

import pandas as pd

from .utils import check_data_columns


def above_percent(
    data: Union[pd.DataFrame, pd.Series, list],
    targets_above: List[int] = [140, 180, 250],
) -> pd.DataFrame:
    """
    Calculate percentage of values above target thresholds.

    The function produces a DataFrame with values equal to the percentage of glucose
    measurements above target values. The output columns correspond to the subject id
    followed by the target values, and the output rows correspond to the subjects.
    The values will be between 0 (no measurements) and 100 (all measurements).

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values, or a list of glucose values
    targets_above : List[float], default=[140, 180, 250]
        List of glucose thresholds. Glucose values from data argument will be compared
        to each value in the targets_above list.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for each target value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Rodbard (2009) Interpretation of continuous glucose monitoring data:
    glycemic variability and quality of glycemic control,
    Diabetes Technology and Therapeutics 11:55-67,
    doi:10.1089/dia.2008.0132.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> above_percent(data)
       id  above_140  above_180  above_250
    0  subject1      100.0       50.0        0.0
    1  subject2       50.0       50.0        0.0

    >>> above_percent(data['gl'], targets_above=[150, 200])
       above_150  above_200
    0       75.0       25.0
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list)):
        # Convert targets to float
        targets_above = [int(t) for t in targets_above]

        # Calculate total non-NA readings
        total_readings = len(data.dropna())
        if total_readings == 0:
            return pd.DataFrame(columns=[f"above_{t}" for t in targets_above])

        # Calculate percentages for each target
        percentages = {}
        for target in targets_above:
            above_count = len(data[data > target])
            percentages[f"above_{target}"] = (above_count / total_readings) * 100

        return pd.DataFrame([percentages])

    # Handle DataFrame input
    data = check_data_columns(data)
    targets_above = [int(t) for t in targets_above]

    # Initialize result list
    result = []

    # Process each subject
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject]
        total_readings = len(subject_data.dropna(subset=["gl"]))

        if total_readings == 0:
            continue

        # Calculate percentages for each target
        percentages = {}
        for target in targets_above:
            above_count = len(subject_data[subject_data["gl"] > target])
            percentages[f"above_{target}"] = (above_count / total_readings) * 100

        percentages["id"] = subject
        result.append(percentages)

    # Convert to DataFrame
    return pd.DataFrame(result)
