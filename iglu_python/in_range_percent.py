from typing import List, Union

import pandas as pd

from .utils import check_data_columns


def in_range_percent(
    data: Union[pd.DataFrame, pd.Series, list],
    target_ranges: List[List[int]] = [[70, 180], [63, 140]],
) -> pd.DataFrame:
    """
    Calculate percentage of values within target ranges.

    The function produces a DataFrame with values equal to the percentage of glucose
    measurements within specified ranges. The output columns correspond to the subject id
    followed by the target ranges, and the output rows correspond to the subjects.
    The values will be between 0 (no measurements) and 100 (all measurements).

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values, or a list of glucose values
    target_ranges : List[List[int]], default=[[70, 180], [63, 140]]
        List of target value ranges. Each range is a list of two values [min, max].
        Default ranges are:
        - [70, 180] mg/dL: recommended for assessing glycemic control in type 1 or type 2 diabetes
        - [63, 140] mg/dL: recommended for assessing glycemic control during pregnancy

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for each target range. If a list of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Rodbard (2009) Interpretation of continuous glucose monitoring data:
    glycemic variability and quality of glycemic control,
    Diabetes Technology and Therapeutics 11:55-67,
    doi:10.1089/dia.2008.0132.

    Battelino et al. (2019) Clinical targets for continuous glucose monitoring data
    interpretation: recommendations from the international consensus on time in range.
    Diabetes Care 42(8):1593-603, doi:10.2337/dci19-0028

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> in_range_percent(data)
       id  in_range_70_180  in_range_63_140
    0  subject1          50.0            50.0
    1  subject2          50.0            50.0

    >>> in_range_percent(data['gl'], target_ranges=[[80, 200]])
       in_range_80_200
    0             75.0
    """
    # Handle Series input
    if isinstance(data, (pd.Series, list)):
        # Calculate total non-NA readings
        total_readings = len(data.dropna())
        if total_readings == 0:
            return pd.DataFrame(
                columns=[f"in_range_{min(r)}_{max(r)}" for r in target_ranges]
            )

        # Calculate percentages for each range
        percentages = {}
        for range_vals in target_ranges:
            min_val, max_val = sorted(range_vals)
            in_range_count = len(data[(data >= min_val) & (data <= max_val)])
            percentages[f"in_range_{min_val}_{max_val}"] = (
                in_range_count / total_readings
            ) * 100

        return pd.DataFrame([percentages])

    data = check_data_columns(data)

    # Initialize result list
    result = []

    # Process each subject
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject]
        total_readings = len(subject_data.dropna(subset=["gl"]))

        if total_readings == 0:
            continue

        # Calculate percentages for each range
        percentages = {}
        for range_vals in target_ranges:
            min_val, max_val = sorted(range_vals)
            in_range_count = len(
                subject_data[
                    (subject_data["gl"] >= min_val) & (subject_data["gl"] <= max_val)
                ]
            )
            percentages[f"in_range_{min_val}_{max_val}"] = (
                in_range_count / total_readings
            ) * 100

        percentages["id"] = subject
        result.append(percentages)

    # Convert to DataFrame
    return pd.DataFrame(result)
