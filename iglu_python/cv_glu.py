"""Calculate Coefficient of Variation (CV) of glucose levels.

This module provides a function to calculate the Coefficient of Variation (CV) of glucose measurements.
CV is a measure of relative variability, calculated as 100 * standard deviation / mean.

References:
    Rodbard (2009) Interpretation of continuous glucose monitoring data:
    glycemic variability and quality of glycemic control,
    Diabetes Technology and Therapeutics 11 .55-67,
    doi:10.1089/dia.2008.0132.
"""

from typing import Union

import numpy as np
import pandas as pd

from .utils import check_data_columns


def cv_glu(data: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> Union[pd.DataFrame, float]:
    """Calculate Coefficient of Variation (CV) of glucose levels.

    The function cv_glu produces CV values in a pandas DataFrame object.

    Args:
        data: DataFrame object with column names "id", "time", and "gl",
              or pandas Series of glucose values.

    Returns:
        If a DataFrame object is passed, then a DataFrame with two columns:
        subject id and corresponding CV value is returned. If a Series of glucose
        values is passed, then a DataFrame with just the CV value is returned.

    Details:
        A DataFrame with 1 row for each subject, a column for subject id and
        a column for CV values is returned. NA glucose values are
        omitted from the calculation of the CV.

        CV (Coefficient of Variation) is calculated by 100 * sd(G) / mean(G)
        Where G is the list of all Glucose measurements for a subject.
    """
    # Handle Series input
    if isinstance(data, (list, np.ndarray, pd.Series)):
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        data = data.dropna()
        if len(data) == 0:
            raise ValueError("No glucose values provided")
        # Calculate CV for Series
        cv_val = 100 * data.std() / data.mean()
        return cv_val

    # Check and prepare data
    data = check_data_columns(data)

    data = data.dropna()
    # Calculate CV for each subject
    out = data.groupby("id").agg(CV=("gl", lambda x: 100 * x.std() / x.mean())).reset_index()

    return out
