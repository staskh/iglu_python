from typing import Union
import warnings

import pandas as pd
import numpy as np

from .utils import check_data_columns


def summary_glu(data: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> pd.DataFrame:
    """
    Calculate summary glucose level
    
    The function summary_glu is a wrapper that produces summary statistics
    for glucose data. Output is a DataFrame object with subject id and the 
    summary values: Minimum, 1st Quartile, Median, Mean, 3rd Quartile and Max.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, list, or np.ndarray
        DataFrame object with column names "id", "time", and "gl",
        or numeric vector/array/list of glucose values.

    Returns
    -------
    pd.DataFrame
        If a DataFrame object is passed, then a DataFrame object with
        a column for subject id and then a column for each summary value is returned. 
        If a vector of glucose values is passed, then a DataFrame object without 
        the subject id is returned.

    Details
    -------
    A DataFrame object with 1 row for each subject, a column for subject id and
    a column for each of summary values is returned. NA glucose values are
    omitted from the calculation of the summary values.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1'] * 4,
    ...     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...                            '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
    ...     'gl': [150, 200, 180, 160]
    ... })
    >>> result = summary_glu(data)
    >>> print(result.columns.tolist())
    ['id', 'Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
    """
    # Handle vector input (Series, list, or numpy array)
    is_vector = False
    
    if isinstance(data, (pd.Series, list, np.ndarray)):
        is_vector = True
        
        # Convert to numpy array for consistent handling
        if isinstance(data, pd.Series):
            glucose_values = data.values
        elif isinstance(data, list):
            glucose_values = np.array(data)
        else:  # numpy array
            glucose_values = data
        
        # Remove NaN values
        glucose_values = glucose_values[~np.isnan(glucose_values)]
        
        if len(glucose_values) == 0:
            raise ValueError("No valid glucose values found")
        
        # Calculate summary statistics
        summary_stats = _calculate_summary_stats(glucose_values)
        
        # Return DataFrame without id column
        return pd.DataFrame([summary_stats])
    
    # Handle DataFrame input
    else:
        # Check data format
        data = check_data_columns(data)
        
        # Filter out missing glucose values and group by id
        result_rows = []
        
        for subject_id in data['id'].unique():
            subject_data = data[data['id'] == subject_id]
            glucose_values = subject_data['gl'].dropna().values
            
            if len(glucose_values) == 0:
                warnings.warn(f"No valid glucose values found for subject {subject_id}")
                # Still include the subject with NaN values
                summary_stats = {
                    'Min.': np.nan,
                    '1st Qu.': np.nan,
                    'Median': np.nan,
                    'Mean': np.nan,
                    '3rd Qu.': np.nan,
                    'Max.': np.nan
                }
            else:
                summary_stats = _calculate_summary_stats(glucose_values)
            
            # Add subject id to the summary
            summary_stats['id'] = subject_id
            result_rows.append(summary_stats)
        
        # Create result DataFrame with id column first
        result_df = pd.DataFrame(result_rows)
        
        # Reorder columns to match R output (id first, then summary stats)
        column_order = ['id', 'Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
        return result_df[column_order]


def _calculate_summary_stats(glucose_values: np.ndarray) -> dict:
    """
    Calculate summary statistics for glucose values.
    
    This mimics R's summary() function output for numeric vectors.
    
    Parameters
    ----------
    glucose_values : np.ndarray
        Array of glucose values (without NaN)
    
    Returns
    -------
    dict
        Dictionary with summary statistics matching R's summary() output
    """
    return {
        'Min.': np.min(glucose_values),
        '1st Qu.': np.percentile(glucose_values, 25),
        'Median': np.median(glucose_values),
        'Mean': np.mean(glucose_values),
        '3rd Qu.': np.percentile(glucose_values, 75),
        'Max.': np.max(glucose_values)
    } 