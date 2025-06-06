from typing import List, Union

import pandas as pd

from .below_percent import below_percent
from .in_range_percent import in_range_percent
from .sd_glu import sd_glu
from .utils import check_data_columns


def cogi(
    data: Union[pd.DataFrame, pd.Series, list],
    targets: List[int] = [70, 180],
    weights: List[float] = [0.5, 0.35, 0.15],
) -> pd.DataFrame:
    """
    Calculate Coefficient of Glucose Irregularity (COGI).

    The function produces COGI values in a DataFrame object. COGI is calculated by combining
    three components with specified weights:
    1. Time in range (between the two target values)
    2. Time below range (below the lower target value)
    3. Glucose variability (standard deviation)

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, list]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values, or a list of glucose values
    targets : List[int], default=[70, 180]
        List of two glucose values for threshold. The lower value is used for determining
        time below range, and both values define the target range.
    weights : List[float], default=[0.5, 0.35, 0.15]
        List of three weights to be applied to time in range, time below range,
        and glucose variability, respectively.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row for each subject, a column for subject id and a column
        for the COGI value. If a list of glucose values is passed, then a DataFrame
        without the subject id is returned.

    References
    ----------
    Leelarathna (2020) Evaluating Glucose Control With a Novel Composite
    Continuous Glucose Monitoring Index,
    Diabetes Technology and Therapeutics 14(2) 277-284,
    doi:10.1177/1932296819838525.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> cogi(data)
       id    COGI
    0  subject1  75.5
    1  subject2  82.3

    >>> cogi(data['gl'], targets=[80, 150], weights=[0.3, 0.6, 0.1])
       COGI
    0  68.9
    """

    def weight_features(
        feature: Union[float, pd.Series, list],
        scale_range: List[float],
        weight: float = 1,
        increasing: bool = False,
    ) -> Union[float, pd.Series, list]:
        """Helper function to weight and scale features. If feature is a Series (or a list), the output is a Series (or list) with the same number of rows (or length) as the input, with values clipped (or "inverse" clipped) so that they are between 0 and 1."""
        if isinstance(feature, pd.Series):
            scaled = (feature - min(scale_range)) / (
                max(scale_range) - min(scale_range)
            )
            if increasing:
                out = scaled.clip(lower=0, upper=1)
            else:
                out = (1 - scaled).clip(lower=0, upper=1)
        elif isinstance(feature, list):
            scaled = [
                (x - min(scale_range)) / (max(scale_range) - min(scale_range))
                for x in feature
            ]
            if increasing:
                out = [min(1, max(0, x)) for x in scaled]
            else:
                out = [min(1, max(0, 1 - x)) for x in scaled]
        else:
            scaled = (feature - min(scale_range)) / (
                max(scale_range) - min(scale_range)
            )
            if increasing:
                out = min(1, max(0, scaled))
            else:
                out = min(1, max(0, 1 - scaled))
        return out * weight

    # Check and prepare data
    is_vector = isinstance(data, (pd.Series, list))
    if not is_vector:
        data = check_data_columns(data)
    targets = sorted([float(t) for t in targets])

    # Calculate components
    ir_df = in_range_percent(data, [targets])
    ir = ir_df["in_range_" + "_".join(map(str, targets))]
    br_df = below_percent(data, targets_below=[targets[0]])
    br = br_df["below_" + str(int(targets[0]))]
    stddev_df = sd_glu(data)
    stddev = stddev_df["SD"]

    # Calculate weighted features
    weighted_features = (
        weight_features(ir, [0, 100], weight=weights[0], increasing=True)
        + weight_features(br, [0, 15], weight=weights[1])
        + weight_features(stddev, [18, 108], weight=weights[2])
    )

    # Create output DataFrame
    out = pd.DataFrame({"COGI": weighted_features * 100})  # Convert to percentage
    if not is_vector:
        out["id"] = stddev_df["id"]
        out = out[["id", "COGI"]]

    return out
