"""Unit tests for CV measures (CVmean and CVsd) calculation."""

import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "cv_measures"


def get_test_scenarios():
    """Get test scenarios for GVP calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for GVP method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_cv_measures_iglu_r_compatible(scenario):
    """Test CV measures calculation against expected results from R implementation."""
    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})


    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Calculate CV measures
    result_df = iglu.cv_measures(df,**kwargs)

    # Compare with expected results
    pd.testing.assert_frame_equal(
        result_df.round(3),
        expected_df.round(3),
        check_dtype=False,  # Don't check dtypes since we might have different numeric types
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_names=True,
        check_datetimelike_compat=True,
        check_categorical=True,
        check_like=True,
        check_freq=True,
        check_flags=True,
        check_exact=False,
        rtol=1e-3,
    )

def test_cv_measures_basic():
    """Test basic CV measures calculation with known glucose values."""
    # Create test data with two days of measurements for one subject
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                           '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])
    data = pd.DataFrame({
        'id': ['1'] * 6,  # 3 measurements per day for 2 days
        'time': time,
        'gl': [100, 120, 110,  # Day 1: mean=110, std=10, CV=9.09
               90, 130, 95]     # Day 2: mean=105, std=21.21, CV=20.75
    })

    # Calculate CV measures
    result = iglu.cv_measures(data)

    # Expected results:
    # CVmean = np.mean([9.09, 20.75]) = 14.92
    # CVsd = np.std([9.09, 20.75],ddof=1) = 8.244
    expected = pd.DataFrame({
        'id': ['1'],
        'CVmean': [14.92],
        'CVsd': [8.244]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_measures_multiple_subjects():
    """Test CV measures calculation with multiple subjects."""
    # Create test data with two subjects, each with two days of measurements
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                           '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00',
                           '2020-01-03 10:00:00', '2020-01-03 10:05:00', '2020-01-03 10:10:00',
                           '2020-01-04 10:00:00', '2020-01-04 10:05:00', '2020-01-04 10:10:00'])

    data = pd.DataFrame({
        'id': ['1'] * 6 + ['2'] * 6,  # 3 measurements per day for 2 days for each subject
        'time': time,
        'gl': [100, 120, 110,  # Subject 1, Day 1: CV=9.09
               90, 130, 95,     # Subject 1, Day 2: CV=20.20
               80, 100, 90,     # Subject 2, Day 1: CV=11.11
               70, 110, 80]     # Subject 2, Day 2: CV=25.00
    })

    # Calculate CV measures
    result = iglu.cv_measures(data)

    # Expected results:
    # Subject 1: CVmean=14.92, CVsd=8.244
    # Subject 2: CVmean=17.565, CVsd=9.127
    expected = pd.DataFrame({
        'id': ['1', '2'],
        'CVmean': [14.92, 17.565],
        'CVsd': [8.244, 9.127]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_measures_missing_values():
    """Test CV measures calculation with missing values."""
    # Create test data with missing values
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                           '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])

    data = pd.DataFrame({
        'id': ['1'] * 6,  # 3 measurements per day for 2 days
        'time': time,
        'gl': [100, np.nan, 110,  # Day 1: CV=7.07 (after interpolation)
               90, 130, np.nan]    # Day 2: CV=28.28 (after interpolation)
    })

    # Calculate CV measures
    result = iglu.cv_measures(data, inter_gap=45)  # Allow interpolation

    # Expected results:
    # CVmean = mean([7.07, 28.28]) = 17.68
    # CVsd = np.std([7.07, 28.28],ddof=1) = 13.419
    expected = pd.DataFrame({
        'id': ['1'],
        'CVmean': [16.223],
        'CVsd': [13.419]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_measures_constant_glucose():
    """Test CV measures calculation with constant glucose values."""
    # Create test data with constant glucose values
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                           '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])

    data = pd.DataFrame({
        'id': ['1'] * 6,  # 3 measurements per day for 2 days
        'time': time,
        'gl': [100, 100, 100,  # Day 1: CV=0
               100, 100, 100]   # Day 2: CV=0
    })

    # Calculate CV measures
    result = iglu.cv_measures(data)

    # Expected results:
    # CVmean = mean([0, 0]) = 0
    # CVsd = std([0, 0]) = 0
    expected = pd.DataFrame({
        'id': ['1'],
        'CVmean': [0.0],
        'CVsd': [0.0]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-3
    )

def test_cv_measures_single_day():
    """Test CV measures calculation with only one day of data."""
    # Create test data with only one day of measurements
    data = pd.DataFrame({
        'id': ['1'] * 3,  # 3 measurements for one day
        'time': pd.date_range('2020-01-01 10:00:00', periods=3, freq='5min'),
        'gl': [100, 120, 110]  # CV=9.09
    })

    # Calculate CV measures
    result = iglu.cv_measures(data)

    # Expected results:
    # CVmean = 9.09 (only one day)
    # CVsd = NaN (can't calculate std with one value)
    expected = pd.DataFrame({
        'id': ['1'],
        'CVmean': [9.09],
        'CVsd': [np.nan]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_measures_empty():
    """Test CV measures calculation with empty data."""
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        iglu.cv_measures(pd.DataFrame(columns=['id', 'time', 'gl']))

def test_cv_measures_custom_dt0():
    """Test CV measures calculation with custom dt0 parameter."""
    # Create test data
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                           '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])

    data = pd.DataFrame({
        'id': ['1'] * 6,  # 3 measurements per day for 2 days
        'time': time,
        'gl': [100, 120, 110,  # Day 1
               90, 130, 95]     # Day 2
    })

    # Calculate CV measures with custom dt0
    result = iglu.cv_measures(data, dt0=5)  # 5-minute intervals

    # The results should be the same as without dt0 since our data is already in 5-minute intervals
    expected = pd.DataFrame({
        'id': ['1'],
        'CVmean': [14.92],
        'CVsd': [8.244]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_measures_series_with_datetime_index():
    """Test CV measures calculation with Series input that has DatetimeIndex."""
    # Create test data with DatetimeIndex
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                          '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])
    data = pd.Series(
        [100, 120, 110,  # Day 1: mean=110, std=10, CV=9.09
         90, 130, 95],   # Day 2: mean=105, std=21.21, CV=20.75
        index=time
    )

    # Calculate CV measures
    result = iglu.cv_measures(data)

    # Expected results:
    # CVmean = np.mean([9.09, 20.75]) = 14.92
    # CVsd = np.std([9.09, 20.75], ddof=1) = 8.244
    expected = {
        'CVmean': 14.92,
        'CVsd': 8.244
    }

    # Compare results
    assert isinstance(result, dict)
    np.testing.assert_allclose(result['CVmean'], expected['CVmean'], rtol=0.001)
    np.testing.assert_allclose(result['CVsd'], expected['CVsd'], rtol=0.001)

def test_cv_measures_series_without_datetime_index():
    """Test CV measures calculation with Series input that doesn't have DatetimeIndex."""
    # Create test data with regular index
    data = pd.Series(
        [100, 120, 110, 90, 130, 95],
        index=range(6)  # Regular integer index instead of DatetimeIndex
    )

    # Attempt to calculate CV measures - should raise ValueError
    with pytest.raises(ValueError, match="Series must have a DatetimeIndex"):
        iglu.cv_measures(data)

def test_cv_measures_series_with_missing_values():
    """Test CV measures calculation with Series input containing missing values."""
    # Create test data with DatetimeIndex and missing values
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                          '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])
    data = pd.Series(
        [100, np.nan, 110,  # Day 1: CV=7.07 (after interpolation)
         90, 130, np.nan],  # Day 2: CV=28.28 (after interpolation)
        index=time
    )

    # Calculate CV measures with interpolation
    result = iglu.cv_measures(data, inter_gap=45)

    # Expected results:
    # CVmean = 16.223
    # CVsd = 13.419
    expected = {
        'CVmean': 16.223,
        'CVsd': 13.419
    }

    # Compare results
    assert isinstance(result, dict)
    np.testing.assert_allclose(result['CVmean'], expected['CVmean'], rtol=0.001)
    np.testing.assert_allclose(result['CVsd'], expected['CVsd'], rtol=0.001)
