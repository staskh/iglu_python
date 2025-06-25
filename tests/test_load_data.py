"""
Unit tests for iglu_python.extension.load_data module.

Tests the functionality of loading CGM data from device-specific files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import the module to test
from iglu_python import load_libre, load_dexcom

@pytest.fixture(scope="module")
def test_data_paths():
    test_data_dir = Path(__file__).parent / "data"
    return {
        'libre_amer_01': test_data_dir / "libre_amer_01.csv",
        'libre_amer_02': test_data_dir / "libre_amer_02.csv",
        'dexcom_eur_01': test_data_dir / "dexcom_eur_01.xlsx",
        'dexcom_eur_02': test_data_dir / "dexcom_eur_02.xlsx",
        'dexcom_eur_03': test_data_dir / "dexcom_eur_03.xlsx",
    }

@pytest.mark.parametrize("key", [
    'libre_amer_01', 'libre_amer_02', 'dexcom_eur_01', 'dexcom_eur_02', 'dexcom_eur_03'
])
def test_files_exist(test_data_paths, key):
    assert test_data_paths[key].exists(), f"Test file not found: {test_data_paths[key]}"

def test_load_libre_amer_01(test_data_paths):
    timeseries = load_libre(str(test_data_paths['libre_amer_01']))
    assert isinstance(timeseries, pd.Series)
    assert isinstance(timeseries.index, pd.DatetimeIndex)
    assert len(timeseries) > 0
    numeric_values = pd.to_numeric(timeseries, errors='coerce').dropna()
    assert all(35 <= val <= 400 for val in numeric_values)
    assert timeseries.index.is_monotonic_increasing
    expected_first_values = [127, 124, 121, 131, 153]
    actual_first_values = pd.to_numeric(timeseries.head(), errors='coerce').dropna().tolist()
    np.testing.assert_array_almost_equal(actual_first_values, expected_first_values, decimal=0)

def test_load_libre_amer_02(test_data_paths):
    timeseries = load_libre(str(test_data_paths['libre_amer_02']))
    assert isinstance(timeseries, pd.Series)
    assert isinstance(timeseries.index, pd.DatetimeIndex)
    assert len(timeseries) > 0
    numeric_values = pd.to_numeric(timeseries, errors='coerce').dropna()
    assert all(35 <= val <= 400 for val in numeric_values)
    assert timeseries.index.is_monotonic_increasing
    expected_first_values = [118, 120, 128, 137, 132]
    actual_first_values = pd.to_numeric(timeseries.head(), errors='coerce').dropna().tolist()
    np.testing.assert_array_almost_equal(actual_first_values, expected_first_values, decimal=0)

def test_load_libre_data_consistency(test_data_paths):
    ts1 = load_libre(str(test_data_paths['libre_amer_01']))
    ts2 = load_libre(str(test_data_paths['libre_amer_02']))
    assert isinstance(ts1, pd.Series)
    assert isinstance(ts2, pd.Series)
    assert isinstance(ts1.index, pd.DatetimeIndex)
    assert isinstance(ts2.index, pd.DatetimeIndex)
    numeric_1 = pd.to_numeric(ts1, errors='coerce').dropna()
    numeric_2 = pd.to_numeric(ts2, errors='coerce').dropna()
    assert all(35 <= val <= 400 for val in numeric_1)
    assert all(35 <= val <= 400 for val in numeric_2)

def test_load_libre_time_format(test_data_paths):
    timeseries = load_libre(str(test_data_paths['libre_amer_01']))
    sample_times = timeseries.index[:5]
    assert all(time.year == 2021 for time in sample_times)
    assert all(time.month == 3 for time in sample_times)

def test_load_libre_no_nan_values(test_data_paths):
    ts1 = load_libre(str(test_data_paths['libre_amer_01']))
    ts2 = load_libre(str(test_data_paths['libre_amer_02']))
    assert not ts1.isna().any()
    assert not ts2.isna().any()

def test_load_dexcom_eur_01(test_data_paths):
    timeseries = load_dexcom(str(test_data_paths['dexcom_eur_01']))
    assert isinstance(timeseries, pd.Series)
    assert isinstance(timeseries.index, pd.DatetimeIndex)
    assert len(timeseries) > 0
    # Convert to numeric and check values are reasonable glucose values (mg/dL)
    numeric_values = pd.to_numeric(timeseries, errors='coerce').dropna()
    assert all(35 <= val <= 400 for val in numeric_values)
    assert timeseries.index.is_monotonic_increasing

def test_load_dexcom_eur_02(test_data_paths):
    timeseries = load_dexcom(str(test_data_paths['dexcom_eur_02']))
    assert isinstance(timeseries, pd.Series)
    assert isinstance(timeseries.index, pd.DatetimeIndex)
    assert len(timeseries) > 0
    # Convert to numeric and check values are reasonable glucose values (mg/dL)
    numeric_values = pd.to_numeric(timeseries, errors='coerce').dropna()
    assert all(35 <= val <= 400 for val in numeric_values)
    assert timeseries.index.is_monotonic_increasing

def test_load_dexcom_eur_03(test_data_paths):
    timeseries = load_dexcom(str(test_data_paths['dexcom_eur_03']))
    assert isinstance(timeseries, pd.Series)
    assert isinstance(timeseries.index, pd.DatetimeIndex)
    assert len(timeseries) > 0
    # Convert to numeric and check values are reasonable glucose values (mg/dL)
    numeric_values = pd.to_numeric(timeseries, errors='coerce').dropna()
    assert all(35 <= val <= 400 for val in numeric_values)
    assert timeseries.index.is_monotonic_increasing

def test_load_dexcom_data_consistency(test_data_paths):
    ts_01 = load_dexcom(str(test_data_paths['dexcom_eur_01']))
    ts_02 = load_dexcom(str(test_data_paths['dexcom_eur_02']))
    ts_03 = load_dexcom(str(test_data_paths['dexcom_eur_03']))
    # All should have the same structure
    assert isinstance(ts_01, pd.Series)
    assert isinstance(ts_02, pd.Series)
    assert isinstance(ts_03, pd.Series)
    assert isinstance(ts_01.index, pd.DatetimeIndex)
    assert isinstance(ts_02.index, pd.DatetimeIndex)
    assert isinstance(ts_03.index, pd.DatetimeIndex)
    # All should have glucose data
    for ts in [ts_01, ts_02, ts_03]:
        numeric_values = pd.to_numeric(ts, errors='coerce').dropna()
        assert len(numeric_values) > 0

def test_load_dexcom_timestamp_format(test_data_paths):
    timeseries = load_dexcom(str(test_data_paths['dexcom_eur_01']))
    # Check that timestamps are in the expected format
    sample_times = timeseries.index[:5]
    # Check that all times are in 2023 (from the test data)
    assert all(time.year == 2023 for time in sample_times)

def test_load_dexcom_glucose_statistics(test_data_paths):
    ts_01 = load_dexcom(str(test_data_paths['dexcom_eur_01']))
    ts_02 = load_dexcom(str(test_data_paths['dexcom_eur_02']))
    ts_03 = load_dexcom(str(test_data_paths['dexcom_eur_03']))
    
    for ts in [ts_01, ts_02, ts_03]:
        # Convert to numeric for statistics
        numeric_values = pd.to_numeric(ts, errors='coerce').dropna()
        assert len(numeric_values) > 0
        assert numeric_values.mean() > 0
        assert numeric_values.std() > 0
        # Check that means are reasonable glucose values (mg/dL)
        assert 80 <= numeric_values.mean() <= 300

def test_load_libre_error_handling():
    with pytest.raises(FileNotFoundError):
        load_libre("nonexistent_file.csv")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
    try:
        with pytest.raises(Exception):
            load_libre(tmp_file_path)
    finally:
        os.unlink(tmp_file_path)

def test_load_dexcom_error_handling():
    with pytest.raises(FileNotFoundError):
        load_dexcom("nonexistent_file.xlsx")
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
    try:
        with pytest.raises(Exception):
            load_dexcom(tmp_file_path)
    finally:
        os.unlink(tmp_file_path)

def test_load_libre_data_statistics(test_data_paths):
    ts1 = load_libre(str(test_data_paths['libre_amer_01']))
    ts2 = load_libre(str(test_data_paths['libre_amer_02']))
    numeric_1 = pd.to_numeric(ts1, errors='coerce')
    numeric_2 = pd.to_numeric(ts2, errors='coerce')
    assert numeric_1.mean() > 0
    assert numeric_2.mean() > 0
    assert numeric_1.std() > 0
    assert numeric_2.std() > 0
    assert 80 <= numeric_1.mean() <= 200
    assert 80 <= numeric_2.mean() <= 200

def test_load_libre_time_interval(test_data_paths):
    timeseries = load_libre(str(test_data_paths['libre_amer_01']))
    time_diffs = timeseries.index.to_series().diff().dropna()
    expected_interval = pd.Timedelta(minutes=15)
    tolerance = pd.Timedelta(minutes=5)
    close_intervals = time_diffs[abs(time_diffs - expected_interval) <= tolerance]
    assert len(close_intervals) / len(time_diffs) > 0.8  # At least 80% should be close 

def test_load_dexcom_time_interval(test_data_paths):
    timeseries = load_dexcom(str(test_data_paths['dexcom_eur_01']))
    time_diffs = timeseries.index.to_series().diff().dropna()
    # Most intervals should be around 5 minutes (300 seconds)
    # Allow some tolerance for missing data points
    expected_interval = pd.Timedelta(minutes=5)
    tolerance = pd.Timedelta(minutes=2)  # Allow some variation
    # Check that most intervals are close to expected
    close_intervals = time_diffs[abs(time_diffs - expected_interval) <= tolerance]
    assert len(close_intervals) / len(time_diffs) > 0.8  # At least 80% should be close

def test_load_libre_numeric_values(test_data_paths):
    timeseries = load_libre(str(test_data_paths['libre_amer_01']))
    # Check that all values are numeric
    assert pd.api.types.is_numeric_dtype(timeseries)
    # Check that there are no NaN values (all should be valid numbers)
    assert not timeseries.isna().any()

def test_load_dexcom_numeric_values(test_data_paths):
    timeseries = load_dexcom(str(test_data_paths['dexcom_eur_01']))
    # Check that all values are numeric
    assert pd.api.types.is_numeric_dtype(timeseries)
    # Check that there are no NaN values (all should be valid numbers)
    assert not timeseries.isna().any()