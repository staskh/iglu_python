from typing import Union

import numpy as np
import pandas as pd

from .episode_calculation import episode_calculation
from .gvp import gvp
from .in_range_percent import in_range_percent
from .mean_glu import mean_glu
from .utils import check_data_columns


def pgs(
    data: Union[pd.DataFrame, pd.Series], dur_length: int = 20, end_length: int = 30
) -> pd.DataFrame|float:
    """
    Calculate Personal Glycemic State (PGS).

    The function produces a DataFrame with values equal to the PGS score for each subject.
    The output columns correspond to the subject id and PGS value, and the output rows
    correspond to the subjects.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values,
        or a numpy array or list of glucose values
        Should only be data for 1 subject. In case multiple subject ids are detected,
        a warning is produced and only 1st subject is used.
    dur_length : int, optional
        Minimum duration in minutes to be considered an episode. Note dur_length should be
        a multiple of the data recording interval otherwise the function will round up to
        the nearest multiple. Default is 20 minutes to match the original PGS definition.
    end_length : int, optional
        Minimum duration in minutes of improved glycemia for an episode to end.
        Default is 30 minutes to match original PGS definition.

    Returns
    -------
    pd.DataFrame|float
        DataFrame with 1 row for each subject, a column for subject id and a column
        for PGS value. If a Series of glucose values is passed, then a float is returned.

    Notes
    -----
    The formula for PGS is as follows, where GVP = glucose variability percentage,
    MG = mean glucose, PTIR = percent time in range, and N54, N70 are the number of
    hypoglycemic episodes per week in the ranges <54 mg/dL and 54 to <70 mg/dL level
    respectively:

    PGS = f(GVP) + g(MG) + h(PTIR) + j(N54, N70)

    where:
    f(GVP) = 1 + 9/(1 + exp(-0.049(GVP - 65.47)))
    g(MG) = 1 + 9(1/(1 + exp(0.1139(MG - 72.08))) + 1/(1 + exp(-0.09195(MG - 157.57))))
    h(PTIR) = 1 + 9/(1 + exp(0.0833(PTIR - 55.04)))
    j(N54, N70) = a(N54) + b(N70)
    a(N54) = 0.5 + 4.5(1 - exp(-0.91093N54))
    b(N70) = 0.5714N70 + 0.625 if N70 <= 7.65, else 5

    References
    ----------
    Hirsch et al. (2017): A Simple Composite Metric for the Assessment of Glycemic
    Status from Continuous Glucose Monitoring Data: Implications for Clinical Practice
    and the Artificial Pancreas
    Diabetes Technol Ther 19(S3) S38-S48, doi:10.1089/dia.2017.0080.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> pgs(data)
       id        PGS
    0  subject1   X.XXX
    1  subject2   X.XXX
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        return pgs_single(data, dur_length, end_length)

    # Handle DataFrame input
    data = check_data_columns(data)
    data.set_index('time', drop=True, inplace=True)

    out = data.groupby('id').agg(
        PGS = ("gl", lambda x: pgs_single(x, dur_length, end_length))
    ).reset_index()
    return out

def pgs_single(gl: pd.Series, dur_length: int = 20, end_length: int = 30) -> float:
    """Calculate PGS for a single subject"""
    # Calculate components
    gvp_val = gvp(gl)
    mean_val = mean_glu(gl)
    ptir_val = in_range_percent(gl, target_ranges=[[70, 180]])['in_range_70_180']

    # Calculate episode components
    eps = episode_calculation(
        gl,
        lv1_hypo=70,
        lv2_hypo=54,
        dur_length=dur_length,
        end_length=end_length,
    )
    n54 = eps["avg_ep_per_day"].iloc[1] * 7  # Convert to weekly episodes
    n70 = eps["avg_ep_per_day"].iloc[5] * 7  # Use lv1 exclusive, not lv1 super set

    # Calculate PGS components
    f_gvp = 1 + (9 / (1 + np.exp(-0.049 * (gvp_val - 65.47))))
    f_ptir = 1 + (9 / (1 + np.exp(0.0833 * (ptir_val - 55.04))))
    f_mg = 1 + 9 * (
        (1 / (1 + np.exp(0.1139 * (mean_val - 72.08))))
        + (1 / (1 + np.exp(-0.09195 * (mean_val - 157.57))))
    )

    f_h54 = 0.5 + 4.5 * (1 - np.exp(-0.91093 * n54))
    f_h70 = 0.5714 * n70 + 0.625 if n70 <= 7.65 else 5

    # Calculate final PGS score
    pgs_score = f_gvp + f_ptir + f_mg + f_h54 + f_h70

    return pgs_score

