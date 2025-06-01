from typing import List, Union

import pandas as pd

from .utils import check_data_columns


def below_percent(
    data: Union[pd.DataFrame, pd.Series, list], targets_below: List[int] = [54, 70]
) -> pd.DataFrame:
    """
    Calculate percentage of values below target thresholds.

    The function produces a DataFrame with values equal to the percentage of glucose
    measurements below target values. The output columns correspond to the subject id
    followed by the target values, and the output rows correspond to the subjects.
    The values will be between 0 (no measurements) and 100 (all measurements).

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values, or a list of glucose values
    targets_below : List[float], default=[54, 70]
        List of glucose thresholds. Glucose values from data argument will be compared
        to each value in the targets_below list.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for each target value. If a Series or a list of glucose values is passed, then a DataFrame
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
    ...     'gl': [50, 60, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> below_percent(data)
       id  below_54  below_70
    0  subject1      50.0     100.0
    1  subject2       0.0       0.0

    >>> below_percent(data['gl'], targets_below=[50, 60])
       below_50  below_60
    0       25.0      50.0
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list)):
        # Convert targets to float
        targets_below = [int(t) for t in targets_below]

        # Calculate total non-NA readings
        total_readings = len(data.dropna())
        if total_readings == 0:
            return pd.DataFrame(columns=[f"below_{t}" for t in targets_below])

        # Calculate percentages for each target
        percentages = {}
        for target in targets_below:
            below_count = len(data[data < target])
            percentages[f"below_{target}"] = (below_count / total_readings) * 100

        return pd.DataFrame([percentages])

    # Handle DataFrame input
    data = check_data_columns(data)
    targets_below = [int(t) for t in targets_below]

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
        for target in targets_below:
            below_count = len(subject_data[subject_data["gl"] < target])
            percentages[f"below_{target}"] = (below_count / total_readings) * 100

        percentages["id"] = subject
        result.append(percentages)

    # Convert to DataFrame
    return pd.DataFrame(result)
