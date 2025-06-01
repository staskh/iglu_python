from typing import Union

import numpy as np
import pandas as pd

from .above_percent import above_percent
from .below_percent import below_percent
from .utils import check_data_columns


def gri(data: Union[pd.DataFrame, pd.Series], tz: str = "") -> pd.DataFrame:
    """
    Calculate Glycemia Risk Index (GRI).

    The function produces a DataFrame with values equal to the glycemia risk index (GRI) metric.
    The output columns are subject id and GRI value. The output rows correspond to subjects.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    tz : str, default=""
        A string specifying the time zone to be used. Empty string means current time zone.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for GRI value. If a Series of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Klonoff et al. (2022) A Glycemia Risk Index (GRI) of Hypoglycemia and Hyperglycemia
    for Continuous Glucose Monitoring Validated by Clinician Ratings.
    J Diabetes Sci Technol
    doi:10.1177/19322968221085273.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> gri(data)
       id    GRI
    0  subject1  25.45
    1  subject2  15.67

    >>> gri(data['gl'])
       GRI
    0  35.43
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame({"GRI": [np.nan]})

        # Get percentages in each range
        below_54 = below_percent(data, targets_below=[54])["below_54"].iloc[0]
        below_70 = below_percent(data, targets_below=[70])["below_70"].iloc[0]
        above_180 = above_percent(data, targets_above=[180])["above_180"].iloc[0]
        above_250 = above_percent(data, targets_above=[250])["above_250"].iloc[0]

        # Calculate GRI
        gri_value = (
            3.0 * below_54
            + 2.4 * (below_70 - below_54)
            + 1.6 * above_250
            + 0.8 * (above_180 - above_250)
        )

        # Threshold at 100%
        gri_value = min(gri_value, 100)

        return pd.DataFrame({"GRI": [gri_value]})

    # Handle DataFrame input
    data = check_data_columns(data, tz=tz)

    # Calculate GRI for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].dropna(subset=["gl"])
        if len(subject_data) == 0:
            continue

        # Get percentages in each range
        below_54 = below_percent(subject_data, targets_below=[54])["below_54"].iloc[0]
        below_70 = below_percent(subject_data, targets_below=[70])["below_70"].iloc[0]
        above_180 = above_percent(subject_data, targets_above=[180])["above_180"].iloc[
            0
        ]
        above_250 = above_percent(subject_data, targets_above=[250])["above_250"].iloc[
            0
        ]

        # Calculate GRI
        gri_value = (
            3.0 * below_54
            + 2.4 * (below_70 - below_54)
            + 1.6 * above_250
            + 0.8 * (above_180 - above_250)
        )

        # Threshold at 100%
        gri_value = min(gri_value, 100)

        result.append({"id": subject, "GRI": gri_value})

    return pd.DataFrame(result)
