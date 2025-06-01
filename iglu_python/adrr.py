import warnings

import numpy as np
import pandas as pd


def adrr(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average daily risk range (ADRR)

    The function `adrr` produces ADRR values in a DataFrame object.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame object with column names "id", "time", and "gl".

    Returns
    -------
    pd.DataFrame
        A DataFrame object with two columns: subject id and corresponding
        ADRR value.

    Details
    -------
    A DataFrame object with 1 row for each subject, a column for subject id and
    a column for ADRR values is returned. `NaN` glucose values are
    omitted from the calculation of the ADRR values.

    ADRR is the average sum of HBGI corresponding to the highest glucose
    value and LBGI corresponding to the lowest glucose value for each day,
    with the average taken over the daily sums. If there are no high glucose or
    no low glucose values, then 0 will be substituted for the HBGI value or the
    LBGI value, respectively, for that day.

    References
    ----------
    Kovatchev et al. (2006) Evaluation of a New Measure of Blood Glucose Variability in,
    Diabetes. Diabetes care 29:2433-2438.
    DOI: 10.2337/dc06-1085

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> import iglu_python as iglu
    >>>
    >>> # Example data
    >>> data = pd.read_csv('tests/data/example_data_1_subject.csv',index_col=0)
    >>> iglu.adrr(data)
    """

    def adrr_multi(data: pd.DataFrame) -> pd.DataFrame:
        """Internal function to calculate ADRR for multiple subjects"""

        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["time"]):
            try:
                data["time"] = pd.to_datetime(data["time"])
            except Exception as e:
                raise ValueError(f"Could not convert 'time' column to datetime: {e}")

        # Extract date from time
        data = data.copy()
        data["date"] = data["time"].dt.date

        # Filter out NaN glucose values
        data_filtered = data.dropna(subset=["gl"])

        if len(data_filtered) == 0:
            warnings.warn("All glucose values are NaN. Returning empty DataFrame.")
            return pd.DataFrame(columns=["id", "ADRR"])

        # Group by id and date, then calculate BGI and daily risk range
        result = (
            data_filtered.groupby(["id", "date"])
            .apply(lambda group: _calculate_daily_risk(group), include_groups=False)
            .reset_index()
            .groupby("id")["drr"]
            .mean()
            .reset_index()
            .rename(columns={"drr": "ADRR"})
        )

        return result

    def _calculate_daily_risk(group: pd.DataFrame) -> pd.Series:
        """Calculate daily risk range for a single day and subject"""

        # Calculate BGI (Blood Glucose Index)
        bgi = (np.log(group["gl"]) ** 1.084) - 5.381

        # Calculate max and min BGI values for the day
        max_bgi = np.maximum(bgi.max(), 0)
        min_bgi = np.minimum(bgi.min(), 0)

        # Calculate risk components
        max_risk = 22.77 * (max_bgi**2)
        min_risk = 22.77 * (min_bgi**2)

        # Daily risk range is the sum of max and min risks
        drr = min_risk + max_risk

        return pd.Series({"drr": drr})

    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")

    required_columns = ["id", "time", "gl"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"Data must contain columns: {required_columns}. "
            f"Missing columns: {missing_columns}"
        )

    if len(data) == 0:
        warnings.warn("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=["id", "ADRR"])

    # Calculate ADRR
    result = adrr_multi(data)

    return result
