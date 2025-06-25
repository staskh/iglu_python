"""Calculate Glucose Management Indicator (GMI).

This module provides functions to calculate GMI values from continuous glucose monitoring data.
GMI is a metric that estimates HbA1c from CGM data using the formula: 3.31 + (0.02392 * mean_glucose)
where mean_glucose is the average glucose value in mg/dL.

References:
    Bergenstal (2018) Glucose Management Indicator (GMI): A New Term for
    Estimating A1C From Continuous Glucose Monitoring
    Hormone and Metabolic Research 41 .2275-2280,
    doi:10.2337/dc18-1581.
"""

from typing import Union

import numpy as np
import pandas as pd

from iglu_python.utils import check_data_columns


def gmi(data: Union[pd.DataFrame, pd.Series, list]) -> float | pd.DataFrame:
    """Calculate GMI (Glucose Management Indicator).

    The function gmi produces GMI values in a pandas DataFrame object.

    Args:
        data: DataFrame object with column names "id", "time", and "gl",
            or numeric vector of glucose values.

    Returns:
        If a DataFrame object is passed, then a DataFrame with two columns:
        subject id and corresponding GMI is returned. If a vector of glucose
        values is passed, then a DataFrame with just the GMI value is returned.

    Note:
        A DataFrame with 1 row for each subject, a column for subject id and
        a column for GMI values is returned. NA glucose values are
        omitted from the calculation of the GMI.

        GMI score is calculated by 3.31 + (0.02392 * mean(G))
        where G is the vector of Glucose Measurements (mg/dL).
    """
    # Handle Series input
    if isinstance(data, (list, np.ndarray, pd.Series)):
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        # Calculate GMI for Series
        gmi_val = 3.31 + (0.02392 * data.mean())
        return gmi_val

    # Check and prepare data
    data = check_data_columns(data)
    getattr(data, "is_vector", False)

    # Calculate GMI for each subject
    out = data.groupby("id").agg(GMI=("gl", lambda x: 3.31 + (0.02392 * x.mean()))).reset_index()

    return out
