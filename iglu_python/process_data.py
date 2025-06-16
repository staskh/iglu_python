import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .utils import localize_naive_timestamp


def process_data(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    id: Optional[str] = None,
    timestamp: Optional[str] = None,
    glu: Optional[str] = None,
    time_parser: Optional[Callable] = None
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
    # Default time parser
    if time_parser is None:
        time_parser = pd.to_datetime

    # Validate input data type
    if not isinstance(data, (pd.DataFrame, pd.Series, list, np.ndarray)):
        raise TypeError("Invalid data type, please use DataFrame, Series, list, or numpy array.")

    # Convert to DataFrame if necessary
    if isinstance(data, (list, np.ndarray)):
        if glu is None and timestamp is None and id is None:
            # Assume it's just glucose values
            data = pd.DataFrame({'gl': data})
        else:
            raise ValueError("Cannot process list/array data with column specifications. Please provide a DataFrame.")

    if isinstance(data, pd.Series):
        if data.index.dtype.kind == 'M':  # datetime index
            data = pd.DataFrame({'time': data.index, 'gl': data.values})
        else:
            data = pd.DataFrame({'gl': data.values})

    # Ensure we have a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Could not convert data to DataFrame")

    # Drop NAs
    data = data.dropna()

    if data.empty:
        raise ValueError("No data remaining after removing NAs")

    # Make column names lowercase for matching
    original_columns = data.columns.tolist()
    data.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in data.columns]

    # Process id column
    if id is None:
        print("No 'id' parameter passed, defaulting id to 1")
        data.insert(0, 'id', pd.Series(['1'] * len(data), dtype='string'))
    else:
        if not isinstance(id, str):
            raise ValueError("User-defined id name must be string.")

        id_lower = id.lower()
        if id_lower not in data.columns:
            warning_msg = (f"Could not find user-defined id argument name '{id}' in dataset. "
                          f"Available columns: {original_columns}")
            warnings.warn(warning_msg, stacklevel=2)

            # Check if there's a column named 'id'
            if 'id' in data.columns:
                raise ValueError("Fix user-defined argument name for id. "
                               "Note: A column in the dataset DOES match the name 'id': "
                               "If this is the correct column, indicate as such in function argument. "
                               "i.e. id = 'id'")
            else:
                raise ValueError(f"Column '{id}' not found in data")

        # Move id column to first position and rename
        id_col = data[id_lower]
        data = data.drop(columns=[id_lower])
        data.insert(0, 'id', id_col.astype('string'))

    # Process timestamp column
    if timestamp is None:
        if 'time' not in data.columns:
            raise ValueError("No timestamp column specified and no 'time' column found")
        timestamp_col = 'time'
    else:
        if not isinstance(timestamp, str):
            raise ValueError("User-defined timestamp name must be string.")

        timestamp_lower = timestamp.lower()
        if timestamp_lower not in data.columns:
            warning_msg = (f"Could not find user-defined timestamp argument name '{timestamp}' in dataset. "
                          f"Available columns: {original_columns}")
            warnings.warn(warning_msg, stacklevel=2)

            # Check if there's a column named 'time'
            if 'time' in data.columns:
                raise ValueError("Fix user-defined argument name for timestamp. "
                               "Note: A column in the dataset DOES match the name 'time': "
                               "If this is the correct column, indicate as such in function argument. "
                               "i.e. timestamp = 'time'")
            else:
                raise ValueError(f"Column '{timestamp}' not found in data")

        timestamp_col = timestamp_lower

    # Move timestamp column to second position and rename
    if 'time' not in data.columns or timestamp_col != 'time':
        time_data = data[timestamp_col]
        if timestamp_col != 'time':
            data = data.drop(columns=[timestamp_col])

        # Parse time
        try:
            time_data = time_parser(time_data)
        except Exception as e:
            raise ValueError(f"Failed to parse times, ensure times are in parsable format. "
                           f"Original error: {str(e)}") from e

        # Insert at position 1 (after id)
        data.insert(1, 'time', time_data)

    # localize time if in naive format
    data["time"] = pd.to_datetime(data["time"]).apply(localize_naive_timestamp)

    # Process glucose column
    if glu is None:
        if 'gl' not in data.columns:
            raise ValueError("No glucose column specified and no 'gl' column found")
        glu_col = 'gl'
    else:
        if not isinstance(glu, str):
            raise ValueError("User-defined glucose name must be string.")

        glu_lower = glu.lower()
        if glu_lower not in data.columns:
            warning_msg = (f"Could not find user-defined glucose argument name '{glu}' in dataset. "
                          f"Available columns: {original_columns}")
            warnings.warn(warning_msg, stacklevel=2)

            # Check if there's a column named 'gl'
            if 'gl' in data.columns:
                raise ValueError("Fix user-defined argument name for glucose. "
                               "Note: A column in the dataset DOES match the name 'gl': "
                               "If this is the correct column, indicate as such in function argument. "
                               "i.e. glu = 'gl'")
            else:
                raise ValueError(f"Column '{glu}' not found in data")

        glu_col = glu_lower

    # Check if glucose values are in mmol/L
    mmol_conversion = False
    if glu and 'mmol/l' in glu.lower():
        mmol_conversion = True

    # Move glucose column to third position and rename
    if 'gl' not in data.columns or glu_col != 'gl':
        gl_data = data[glu_col]
        if glu_col != 'gl':
            data = data.drop(columns=[glu_col])

        # Convert to numeric
        try:
            gl_data = pd.to_numeric(gl_data, errors='coerce')
        except Exception as e:
            raise ValueError(f"Failed to convert glucose values to numeric: {str(e)}") from e

        # Convert mmol/L to mg/dL if needed
        if mmol_conversion:
            gl_data = gl_data * 18

        # Insert at position 2 (after id and time)
        data.insert(2, 'gl', gl_data)

    # Validation warnings
    if data['gl'].min() < 20:
        warnings.warn("Minimum glucose reading below 20. Data may not be cleaned.", stacklevel=2)

    if data['gl'].max() > 500:
        warnings.warn("Maximum glucose reading above 500. Data may not be cleaned.", stacklevel=2)

    # Keep only the three required columns in correct order
    data = data[['id', 'time', 'gl']]

    # Drop rows with NaN glucose values
    data = data.dropna(subset=['gl'])

    if data.empty:
        raise ValueError("No valid data remaining after processing")

    return data
