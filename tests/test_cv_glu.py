"""Unit tests for CV (Coefficient of Variation) calculation."""

import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "cv_glu"

def get_test_scenarios():
    """Get test scenarios for summary_glu calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for summary_glu method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_cv_glu_iglu_r_compatible(scenario):
    """Test CV calculation against expected results from R implementation."""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])


    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})

    # Calculate CV
    result_df = iglu.cv_glu(df,**kwargs)

    assert result_df is not None

    # Compare DataFrames with precision to 0.001 for numeric columns
    pd.testing.assert_frame_equal(
        result_df,
        expected_df,
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
        rtol=0.001,
    )

def test_cv_glu_basic():
    """Test basic CV calculation with known glucose values."""
    # Create test data with two subjects
    data = pd.DataFrame({
        'id': ['1'] * 3 + ['2'] * 3,
        'time': pd.date_range('2020-01-01', periods=6, freq='5min'),
        'gl': [100, 120, 110, 90, 130, 95]
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected results:
    # Subject 1: CV = 100 * np.std([100, 120, 110],ddof=1) / np.mean([100, 120, 110]) ≈ 9.09
    # Subject 2: CV = 100 * np.std([90, 130, 95],ddof=1) / np.mean([90, 130, 95]) ≈ 20.75.75
    expected = pd.DataFrame({
        'id': ['1', '2'],
        'CV': [9.09, 20.75]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2  # Allow for small numerical differences
    )

def test_cv_glu_series():
    """Test CV calculation with pandas Series input."""
    # Create test data
    data = pd.Series([100, 120, 110, 90, 130, 95])

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected result: CV = 100 * std([100, 120, 110, 90, 130, 95],ddof=1) / mean([100, 120, 110, 90, 130, 95]) ≈ 14.14
    expected = 14.33

    np.testing.assert_allclose(result, expected, rtol=0.001)

def test_cv_glu_empty():
    """Test CV calculation with empty data."""
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        iglu.cv_glu(pd.DataFrame(columns=['id', 'time', 'gl']))

    # Test with empty Series
    with pytest.raises(ValueError):
        iglu.cv_glu(pd.Series([]))

def test_cv_glu_constant_glucose():
    """Test CV calculation with constant glucose values."""
    # Create test data with constant glucose
    data = pd.DataFrame({
        'id': ['1'] * 3,
        'time': pd.date_range('2020-01-01', periods=3, freq='5min'),
        'gl': [100, 100, 100]
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected result: CV = 0 (since std = 0)
    expected = pd.DataFrame({
        'id': ['1'],
        'CV': [0.0]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-3
    )

def test_cv_glu_missing_values():
    """Test CV calculation with missing values."""
    # Create test data with missing values
    data = pd.DataFrame({
        'id': ['1'] * 4,
        'time': pd.date_range('2020-01-01', periods=4, freq='5min'),
        'gl': [100, np.nan, 120, 110]
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected result: CV = 100 * np.std([100, 120, 110],ddof=1) / np.mean([100, 120, 110]) ≈ 9.0909
    expected = pd.DataFrame({
        'id': ['1'],
        'CV': [9.0909]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_glu_extreme_values():
    """Test CV calculation with extreme glucose values."""
    # Create test data with extreme values
    data = pd.DataFrame({
        'id': ['1'] * 3,
        'time': pd.date_range('2020-01-01', periods=3, freq='5min'),
        'gl': [40, 400, 40]  # Very low and very high values
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected result: CV = 100 * std([40, 400, 40],ddof=1) / mean([40, 400, 40]) ≈ 129.90
    expected = pd.DataFrame({
        'id': ['1'],
        'CV': [129.90]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_glu_single_subject():
    """Test CV calculation with a single subject."""
    # Create test data for one subject
    data = pd.DataFrame({
        'id': ['1'] * 5,
        'time': pd.date_range('2020-01-01', periods=5, freq='5min'),
        'gl': [120, 118, 122, 119, 121]  # Small variations around 120
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected result: CV = 100 * std([120, 118, 122, 119, 121],ddof=1) / mean([120, 118, 122, 119, 121]) ≈ 1.317
    expected = pd.DataFrame({
        'id': ['1'],
        'CV': [1.317]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_glu_uneven_measurements():
    """Test CV calculation with subjects having different numbers of measurements."""
    # Create test data with two subjects having different numbers of measurements
    data = pd.DataFrame({
        'id': ['1'] * 3 + ['2'] * 5,  # Subject 1 has 3 measurements, Subject 2 has 5
        'time': pd.date_range('2020-01-01', periods=8, freq='5min'),
        'gl': [100, 120, 110,  # Subject 1
               90, 130, 95, 125, 105]  # Subject 2
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected results:
    # Subject 1: CV = 100 * std([100, 120, 110],ddof=1) / mean([100, 120, 110]) ≈ 9.0909
    # Subject 2: CV = 100 * std([90, 130, 95, 125, 105],ddof=1) / mean([90, 130, 95, 125, 105]) ≈ 16.3472
    expected = pd.DataFrame({
        'id': ['1', '2'],
        'CV': [9.0909, 16.3472]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )

def test_cv_glu_mixed_missing():
    """Test CV calculation with mixed missing values across subjects."""
    # Create test data with different patterns of missing values
    data = pd.DataFrame({
        'id': ['1'] * 3 + ['2'] * 3 + ['3'] * 3,
        'time': pd.date_range('2020-01-01', periods=9, freq='5min'),
        'gl': [100, np.nan, 110,  # Subject 1: one missing
               90, 130, np.nan,    # Subject 2: one missing
               np.nan, np.nan, 95] # Subject 3: two missing
    })

    # Calculate CV
    result = iglu.cv_glu(data)

    # Expected results:
    # Subject 1: CV = 100 * std([100, 110]) / mean([100, 110]) ≈ 4.76
    # Subject 2: CV = 100 * std([90, 130]) / mean([90, 130]) ≈ 18.18
    # Subject 3: CV = 0 (only one non-missing value)
    expected = pd.DataFrame({
        'id': ['1', '2', '3'],
        'CV': [6.73435, 25.712, np.nan]
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        rtol=1e-2
    )
