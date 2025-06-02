
import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns , gd2d_to_df


def auc(data: pd.DataFrame, tz: str = "") -> pd.DataFrame:
    """
    Calculate Area Under Curve (AUC) for glucose measurements.

    The function produces hourly average AUC for each subject. AUC is calculated
    for every hour using the trapezoidal rule, then hourly average AUC is calculated
    for each 24 hour period, then the mean of hourly average AUC across all 24 hour
    periods is returned as overall hourly average AUC.

    AUC is calculated using the formula: (dt0/60) * ((gl[2:length(gl)] + gl[1:(length(gl)-1)])/2),
    where dt0/60 is the frequency of the cgm measurements in hours and gl are the glucose values.

    This formula is based off the Trapezoidal Rule: 
    (time[2]-time[1] * ((glucose[1]+glucose[2])/2)).


    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    tz : str, default=""
        Time zone to be used. Empty string means current time zone, "GMT" means UTC.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier
        - hourly_auc: hourly average AUC value

    References
    ----------
    Danne et al. (2017) International Consensus on Use of Continuous Glucose Monitoring,
    Diabetes Care 40:1631-1640,
    doi:10.2337/dc17-1600.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:10:00', '2020-01-01 00:00:00',
    ...                            '2020-01-01 00:05:00']),
    ...     'gl': [150, 155, 160, 140, 145]
    ... })
    >>> auc(data)
       id  hourly_auc
    0  subject1      155.0
    1  subject2      142.5
    """
    # Check data format and convert time to datetime
    data = check_data_columns(data)

    def auc_single(subject_data: pd.DataFrame) -> float:
        """Calculate AUC for a single subject"""
        # Get interpolated data using CGMS2DayByDay
        gd2d, actual_dates, dt0 = CGMS2DayByDay(subject_data, tz=tz)

        # Convert gd2d to DataFrame
        hourly_data = gd2d_to_df(gd2d, actual_dates, dt0)
        # Add hour column by rounding time to nearest hour
        hourly_data['hour'] = hourly_data['time'].dt.floor('h')

        hourly_data['gl_next'] = hourly_data['gl'].shift(-1)

        # Calculate AUC for each hour using trapezoidal rule (mg*min/dL)
        hourly_auc = hourly_data.groupby("hour").apply(
            lambda x: np.nansum(
                (dt0/60)*(x["gl"].values + x["gl_next"].values) / 2
            ),
            include_groups=False
        )
        # 0 mean no data in this hour, replace with nan
        hourly_auc = hourly_auc.replace(0, np.nan)

        hourly_avg = hourly_auc.mean(skipna=True)
        # Return mean of daily hourly averages
        return hourly_avg

    # Process each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject]
        hourly_auc = auc_single(subject_data)
        result.append({"id": subject, "hourly_auc": hourly_auc})

    # Convert to DataFrame
    return pd.DataFrame(result)
