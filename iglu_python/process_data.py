import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .utils import localize_naive_timestamp


def _validate_input_data(data: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> None:
    """Validate input data type"""
    if not isinstance(data, (pd.DataFrame, pd.Series, list, np.ndarray)):
        raise TypeError("Invalid data type, please use DataFrame, Series, list, or numpy array.")


def _convert_to_dataframe(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    glu: Optional[str],
    timestamp: Optional[str],
    id: Optional[str],
) -> pd.DataFrame:
    """Convert input data to DataFrame"""
    if isinstance(data, (list, np.ndarray)):
        if all(param is None for param in [glu, timestamp, id]):
            return pd.DataFrame({"gl": data})
        raise ValueError("Cannot process list/array data with column specifications. Please provide a DataFrame.")

    if isinstance(data, pd.Series):
        if data.index.dtype.kind == "M":  # datetime index
            return pd.DataFrame({"time": data.index, "gl": data.values})
        return pd.DataFrame({"gl": data.values})

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Could not convert data to DataFrame")
    return data


def _find_column(
    data: pd.DataFrame, column_name: Optional[str], default_name: str, original_columns: list, param_name: str
) -> str:
    """Find and validate column name"""
    if column_name is None:
        if default_name not in data.columns:
            raise ValueError(f"No {param_name} column specified and no '{default_name}' column found")
        return default_name

    if not isinstance(column_name, str):
        raise ValueError(f"User-defined {param_name} name must be string.")

    column_lower = column_name.lower()
    if column_lower not in data.columns:
        warnings.warn(
            f"Could not find user-defined {param_name} argument name '{column_name}' in dataset. "
            f"Available columns: {original_columns}",
            stacklevel=2,
        )

        if default_name in data.columns:
            raise ValueError(
                f"Fix user-defined argument name for {param_name}. "
                f"Note: A column in the dataset DOES match the name '{default_name}': "
                f"If this is the correct column, indicate as such in function argument. "
                f"i.e. {param_name} = '{default_name}'"
            )
        else:
            raise ValueError(f"Column '{column_name}' not found in data")

    return column_lower


def _process_id_column(data: pd.DataFrame, id: Optional[str], original_columns: list) -> pd.DataFrame:
    """Process and validate ID column"""
    if id is None:
        print("No 'id' parameter passed, defaulting id to 1")
        data.insert(0, "id", pd.Series(["1"] * len(data), dtype="string"))
        return data

    id_col = _find_column(data, id, "id", original_columns, "id")
    id_data = data[id_col]
    data = data.drop(columns=[id_col])
    data.insert(0, "id", id_data.astype("string"))
    return data


def _process_timestamp_column(
    data: pd.DataFrame, timestamp: Optional[str], original_columns: list, time_parser: Callable
) -> pd.DataFrame:
    """Process and validate timestamp column"""
    timestamp_col = _find_column(data, timestamp, "time", original_columns, "timestamp")

    if "time" not in data.columns or timestamp_col != "time":
        time_data = data[timestamp_col]
        if timestamp_col != "time":
            data = data.drop(columns=[timestamp_col])

        try:
            time_data = time_parser(time_data)
        except Exception as e:
            raise ValueError(
                f"Failed to parse times, ensure times are in parsable format. Original error: {str(e)}"
            ) from e

        data.insert(1, "time", time_data)

    data["time"] = pd.to_datetime(data["time"]).apply(localize_naive_timestamp)
    return data


def _process_glucose_column(data: pd.DataFrame, glu: Optional[str], original_columns: list) -> pd.DataFrame:
    """Process and validate glucose column"""
    glu_col = _find_column(data, glu, "gl", original_columns, "glucose")

    # Check if glucose values are in mmol/L
    mmol_conversion = glu and "mmol/l" in glu.lower()

    if "gl" not in data.columns or glu_col != "gl":
        gl_data = data[glu_col]
        if glu_col != "gl":
            data = data.drop(columns=[glu_col])

        try:
            gl_data = pd.to_numeric(gl_data, errors="coerce")
        except Exception as e:
            raise ValueError(f"Failed to convert glucose values to numeric: {str(e)}") from e

        if mmol_conversion:
            gl_data = gl_data * 18

        data.insert(2, "gl", gl_data)

    return data


def _validate_glucose_values(data: pd.DataFrame) -> None:
    """Validate glucose values and issue warnings if needed"""
    if data["gl"].min() < 20:
        warnings.warn("Minimum glucose reading below 20. Data may not be cleaned.", stacklevel=2)
    if data["gl"].max() > 500:
        warnings.warn("Maximum glucose reading above 500. Data may not be cleaned.", stacklevel=2)


def process_data(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    id: Optional[str] = None,
    timestamp: Optional[str] = None,
    glu: Optional[str] = None,
    time_parser: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Data Pre-Processor

    A helper function to assist in pre-processing the user-supplied input data
    for use with other functions. This function ensures that the returned data
    will be compatible with every function within the iglu package. All NAs
    will be removed.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, list, or np.ndarray
        User-supplied dataset containing continuous glucose monitor data. Must
        contain data for time and glucose readings at a minimum. Accepted
        formats are DataFrame, Series, list, or numpy array.
    id : str, optional
        Column name (string) corresponding to subject id column.
        If no value is passed, an id of 1 will be assigned to the data.
    timestamp : str, optional
        Column name (string) corresponding to time values in data. The dates
        can be in any format parsable by pd.to_datetime, or any format accepted
        by the parser passed to time_parser.
    glu : str, optional
        Column name (string) corresponding to glucose values, mg/dL
    time_parser : callable, optional
        Function used to convert datetime strings to time objects. Defaults to
        pd.to_datetime. If your times are in a format not parsable by
        pd.to_datetime, you can pass a custom parsing function.

    Returns
    -------
    pd.DataFrame
        A processed DataFrame object with columns "id", "time", and "gl" that
        cooperates with every other function within the iglu package. All NAs
        will be removed.

    Details
    -------
    If "mmol/l" appears in the glucose column name, the glucose values will be
    multiplied by 18 to convert to mg/dL.

    Raises
    ------
    TypeError
        If data is not in a supported format
    ValueError
        If required columns are not found or cannot be processed

    Notes
    -----
    Based on John Schwenck's data_process for his bp package and
    David Buchanan's R implementation.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'subject_id': ['A', 'A', 'B', 'B'],
    ...     'datetime': ['2020-01-01 10:00:00', '2020-01-01 10:05:00',
    ...                  '2020-01-01 10:00:00', '2020-01-01 10:05:00'],
    ...     'glucose': [120, 130, 110, 125]
    ... })
    >>> processed = process_data(data, id='subject_id', timestamp='datetime', glu='glucose')
    >>> print(processed.columns.tolist())
    ['id', 'time', 'gl']
    """
    time_parser = time_parser or pd.to_datetime

    # Validate and convert input data
    _validate_input_data(data)
    data = _convert_to_dataframe(data, glu, timestamp, id)
    data = data.dropna()

    if data.empty:
        raise ValueError("No data remaining after removing NAs")

    # Normalize columns and process
    original_columns = data.columns.tolist()
    data.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in data.columns]

    data = _process_id_column(data, id, original_columns)
    data = _process_timestamp_column(data, timestamp, original_columns, time_parser)
    data = _process_glucose_column(data, glu, original_columns)
    _validate_glucose_values(data)

    # Final cleanup
    data = data[["id", "time", "gl"]].dropna(subset=["gl"])
    if data.empty:
        raise ValueError("No valid data remaining after processing")

    return data
