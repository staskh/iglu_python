import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "adrr"

def get_test_scenarios():
    """Get test scenarios for ADRR calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for ADRR method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]

@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_adrr_iglu_r_compatible(scenario):
    """Test ADRR calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.adrr(df, **kwargs)

    assert result_df is not None

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})


    # Compare DataFrames with precision to 0.001 for numeric columns
    pd.testing.assert_frame_equal(
        result_df.round(3),
        expected_df.round(3),
        check_dtype=False,  # Don't check dtypes since we might have different numeric types
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_names=True,
        check_exact=True,
        check_datetimelike_compat=True,
        check_categorical=True,
        check_like=True,
        check_freq=True,
        check_flags=True,
    )

def test_adrr_series_with_datetime_index():
    """Test ADRR calculation with Series input that has DatetimeIndex."""
    # Create test data with DatetimeIndex
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                          '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])
    data = pd.Series(
        [100, 120, 110,  # Day 1: LBGI=0.5, HBGI=0.8
         90, 130, 95],   # Day 2: LBGI=0.7, HBGI=1.2
        index=time
    )

    # Calculate ADRR
    result = iglu.adrr(data)

    # Expected results:
    # Day 1: LBGI=0.5, HBGI=0.8, Risk=1.3
    # Day 2: LBGI=0.7, HBGI=1.2, Risk=1.9
    # ADRR = mean([1.3, 1.9]) = 1.6
    expected = 1.538552
    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)

def test_adrr_series_without_datetime_index():
    """Test ADRR calculation with Series input that doesn't have DatetimeIndex."""
    # Create test data with regular index
    data = pd.Series(
        [100, 120, 110, 90, 130, 95],
        index=range(6)  # Regular integer index instead of DatetimeIndex
    )

    # Attempt to calculate ADRR - should raise ValueError
    with pytest.raises(ValueError, match="Series must have a DatetimeIndex"):
        iglu.adrr(data)

def test_adrr_series_with_missing_values():
    """Test ADRR calculation with Series input containing missing values."""
    # Create test data with DatetimeIndex and missing values
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                          '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])
    data = pd.Series(
        [100, np.nan, 110,  # Day 1: LBGI=0.5, HBGI=0.8 (after interpolation)
         90, 130, np.nan],  # Day 2: LBGI=0.7, HBGI=1.2 (after interpolation)
        index=time
    )

    # Calculate ADRR with interpolation
    result = iglu.adrr(data)

    # Expected results:
    # Day 1: LBGI=0.5, HBGI=0.8, Risk=0.48
    # Day 2: LBGI=0.7, HBGI=1.2, Risk=2.45
    # ADRR = mean([0.48, 2.45]) = 1.466489
    expected = 1.466489

    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)
