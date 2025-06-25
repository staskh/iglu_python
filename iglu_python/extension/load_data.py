"""
This module is to load CGM timeseries from device specific files.
It is inspired by https://github.com/cafoala/diametrics/blob/main/src/diametrics/transform.py
"""

from pathlib import Path
import pandas as pd


def load_libre(file_path: str) -> pd.Series:
    """
    Load Libre timeseries from file.

    Parameters
    ----------
    file_path : str
        Path to the Libre device file.

    Returns
    -------
    pd.Series
        Series with datetime index and glucose values.(in mg/dL)

    Examples
    --------
    >>> load_libre("tests/data/libre_amer_01.csv")
    """
    df = _open_file(file_path)

    # Set third row as column headers
    df.columns = df.iloc[2]
    # Drop top rows
    df = df.iloc[3:]
    df.reset_index(inplace=True, drop=True)
    # Keep important columns based on column names
    convert = False
    if 'Historic Glucose(mmol/L)' in df.columns:
        df = df.loc[:, ('Meter Timestamp', 'Historic Glucose(mmol/L)', 'Scan Glucose(mmol/L)')]
        format = '%d-%m-%Y %H:%M'
        convert = True
    elif 'Historic Glucose(mg/dL)' in df.columns:
        df = df.loc[:, ('Meter Timestamp', 'Historic Glucose(mg/dL)', 'Scan Glucose(mg/dL)')]
        format = '%m-%d-%Y %H:%M'
    elif 'Historic Glucose mmol/L' in df.columns:
        df = df.loc[:, ('Device Timestamp', 'Historic Glucose mmol/L', 'Scan Glucose mmol/L')]
        format = '%d-%m-%Y %I:%M %p' 
        convert = True
    else:
        df = df = df.loc[:, ('Device Timestamp', 'Historic Glucose mg/dL', 'Scan Glucose mg/dL')]
        format = '%m-%d-%Y %I:%M %p'
    # Rename columns
    df.columns = ['time', 'glc', 'scan_glc']

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], format=format)

    # Convert glucose values to numeric
    df['glc'] = pd.to_numeric(df['glc'], errors='coerce')
    
    # convert to mg/dL if needed
    if convert:
        df['glc'] = df['glc'] * 18.01559    

    # Drop NaN values and sort by 'time'
    df = df.dropna(subset=['time', 'glc']).sort_values('time').reset_index(drop=True)

    # convert into timeseries
    timeseries = df.set_index('time')['glc']

    return timeseries




def load_dexcom(file_path: str) -> pd.Series:
    """
    Load Dexcom timeseries from file.

    Parameters
    ----------
    file_path : str
        Path to the Dexcom device file.

    Returns
    -------
    pd.Series
        Series with datetime index and glucose values (in mg/dL)

    Examples
    --------
    >>> load_dexcom("tests/data/dexcom_eur_01.xlsx")
    """
    df = _open_file(file_path)

    # Set first row as column headers
    df.columns = df.iloc[0]
    # Drop top rows
    df = df.iloc[1:]
    df.reset_index(inplace=True, drop=True)
    
    # Find timestamp column
    timestamp_cols = [col for col in df.columns if 'Timestamp' in str(col)]
    if not timestamp_cols:
        raise ValueError("No timestamp column found in Dexcom data")
    timestamp_col = timestamp_cols[0]
    
    # Find glucose column
    glucose_cols = [col for col in df.columns if 'Glucose' in str(col)]
    if not glucose_cols:
        raise ValueError("No glucose column found in Dexcom data")
    glucose_col = glucose_cols[0]
    
    # Check if conversion is needed (mmol/L to mg/dL)
    convert = False
    if 'mmol/L' in str(glucose_col):
        convert = True
    
    # Select relevant columns
    df = df.loc[:, [timestamp_col, glucose_col]]
    
    # Rename columns
    df.columns = ['time', 'glc']
    
    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Convert glucose values to numeric
    df['glc'] = pd.to_numeric(df['glc'], errors='coerce')
    
    # Convert to mg/dL if needed
    if convert:
        df['glc'] = df['glc'] * 18.01559
    
    # Drop NaN values and sort by 'time'
    df = df.dropna(subset=['time', 'glc']).sort_values('time').reset_index(drop=True)
    
    # Convert into timeseries
    timeseries = df.set_index('time')['glc']
    
    return timeseries



def _open_file(filepath: str) -> pd.DataFrame:
    """
    Open a file and read its contents into a pandas DataFrame.

    Args:
        filepath (str): The path to the file.

    Returns:
        pandas.DataFrame: The DataFrame containing the file data.

    Raises:
        Exception: If an error occurs while reading the file.
    """
    # TODO: handle S3 path

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    
    # Get file extension using basename
    extension = Path(filepath).suffix
    
    try:
        if extension == '.csv':
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(filepath, header=None, names=[i for i in range(0, 20)])
        elif extension == '.xls' or extension == '.xlsx':
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(filepath, header=None, names=[i for i in range(0, 20)])
        elif extension == '.txt' or extension == '.tsv':
            # Assume that the user uploaded a text file
            df = pd.read_table(filepath, header=None, names=[i for i in range(0, 20)])
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {filepath}") from e