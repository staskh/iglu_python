"""Calculate Coefficient of Variation subtypes (CVmean and CVsd).

This module provides functions to calculate two types of Coefficient of Variation measures:
1. CVmean: Mean of daily coefficient of variations
2. CVsd: Standard deviation of daily coefficient of variations

References:
    Umpierrez, et.al. (2018) Glycemic Variability: How to Measure and Its Clinical
    Implication for Type 2 Diabetes
    The American Journal of Medical Sciences 356 .518-527,
    doi:10.1016/j.amjms.2018.09.010.
"""

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns


def cv_measures(data, dt0=None, inter_gap=45, tz="")->pd.DataFrame|dict[str:float]:
    """Calculate Coefficient of Variation subtypes (CVmean and CVsd).

    The function cv_measures produces CV subtype values in a pandas DataFrame object.

    Args:
        data: DataFrame object with column names "id", "time", and "gl"
        dt0: The time frequency for interpolation in minutes. If None, will match the CGM meter's frequency
        inter_gap: The maximum allowable gap (in minutes) for interpolation. Default is 45
        tz: String name of timezone. Default is ""

    Returns:
        A DataFrame with three columns: subject id and corresponding CV subtype values (CVmean and CVsd)

    Details:
        A DataFrame with 1 row for each subject, a column for subject id and
        a column for each CV subtype value is returned.

        Missing values will be linearly interpolated when close enough to non-missing values.

        1. CVmean:
           Calculated by first taking the coefficient of variation of each day's glucose measurements,
           then taking the mean of all the coefficient of variations. That is, for x
           days we compute cv_1 ... cv_x daily coefficient of variations and calculate
           1/x * sum(cv_i)

        2. CVsd:
           Calculated by first taking the coefficient of variation of each day's glucose measurements,
           then taking the standard deviation of all the coefficient of variations. That is, for d
           days we compute cv_1 ... cv_d daily coefficient of variations and calculate
           std([cv_1, cv_2, ... cv_d])
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")

        results_dict = _calculate_series_cv(data, dt0=dt0, inter_gap=inter_gap, tz=tz)
        return results_dict

    # Check and prepare data
    data = check_data_columns(data)


    # Process each subject
    results = []
    for subject_id in data['id'].unique():
        subject_data = data[data['id'] == subject_id]

        results_dict = _calculate_series_cv(subject_data, dt0=dt0, inter_gap=inter_gap, tz=tz)

        results.append({
            'id': subject_id,
            'CVmean': results_dict['CVmean'],
            'CVsd': results_dict['CVsd']
        })

    return pd.DataFrame(results)

def _calculate_series_cv(subject_data: pd.DataFrame|pd.Series, dt0=None, inter_gap=45, tz="") -> dict[str:float]:
    """Calculate CV for series/single subject input"""

    # Convert to day-by-day format
    gd2d,active_days,dt0 = CGMS2DayByDay(subject_data, dt0=dt0, inter_gap=inter_gap, tz=tz)

    # gd2d is two dimensional array - 1st dimension is day, 2nd dimension is time point
    # active_days is a list of days that have at least 2 non-missing values
    # dt0 is the time frequency for interpolation in minutes

    # calculate devioation and median for each day
    daily_deviations = np.apply_along_axis(np.nanstd, 1, gd2d,ddof=1)
    daily_mean = np.apply_along_axis(np.nanmean, 1, gd2d)

    cv = daily_deviations *100 / daily_mean

    # calculate mean of daily deviations
    cv_mean = np.nanmean(cv)
    cv_sd = np.nanstd(cv,ddof=1)

    return {
        'CVmean': cv_mean,
        'CVsd': cv_sd
    }
