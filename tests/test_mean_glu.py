import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu
from iglu_python.mean_glu import mean_glu

method_name = "mean_glu"


def get_test_scenarios():
    """Get test scenarios for mean glucose calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])

    # Filter scenarios for mean glucose method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_mean_glu_iglu_r_compatible(scenario):
    """Test mean glucose calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    # pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})


    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.mean_glu(df, **kwargs)

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
        rtol=1e-3,
    )


def test_mean_glu_basic():
    """Test basic mean_glu calculation with known values."""
    # Create test data with known glucose values
    data = pd.DataFrame(
        {
            "id": [
                "subject1",
                "subject1",
                "subject1",
                "subject2",
                "subject2",
                "subject2",
            ],
            "time": pd.date_range(start="2020-01-01", periods=6, freq="5min"),
            "gl": [150, 200, 180, 130, 190, 160],
        }
    )

    # Calculate mean_glu
    result = mean_glu(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "mean" in result.columns
    assert len(result) == 2  # Two subjects

    # Check that mean values are correct
    subject1_mean = result[result["id"] == "subject1"]["mean"].iloc[0]
    subject2_mean = result[result["id"] == "subject2"]["mean"].iloc[0]
    assert abs(subject1_mean - 176.67) < 0.01  # 176.67 is the expected mean
    assert abs(subject2_mean - 160.0) < 0.01  # 160.0 is the expected mean


def test_mean_glu_series_input():
    """Test mean_glu calculation with Series input."""
    # Create test data as Series
    data = pd.Series([150, 200, 180, 130, 190, 160])

    # Calculate mean_glu
    result = mean_glu(data)

    # Check output format
    assert isinstance(result, (float,np.float64))

    # Check that mean value is correct
    np.testing.assert_allclose(result, 168.33, rtol=0.001)  # 168.33 is the expected mean


def test_mean_glu_empty_data():
    """Test mean_glu calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=["id", "time", "gl"])

    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        mean_glu(data)


def test_mean_glu_missing_values():
    """Test mean_glu calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, np.nan, 180],
        }
    )

    # Calculate mean_glu
    result = mean_glu(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "mean" in result.columns
    assert len(result) == 1

    # Check that mean value is correct (should ignore NaN)
    assert abs(result["mean"].iloc[0] - 165.0) < 0.01  # 165.0 is the expected mean


def test_mean_glu_constant_values():
    """Test mean_glu calculation with constant values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, 150, 150],  # All values are the same
        }
    )

    # Calculate mean_glu
    result = mean_glu(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "mean" in result.columns
    assert len(result) == 1

    # Check that mean value is correct
    assert result["mean"].iloc[0] == 150.0


def test_mean_glu_multiple_subjects():
    """Test mean_glu calculation with multiple subjects."""
    data = pd.DataFrame(
        {
            "id": [
                "subject1",
                "subject1",
                "subject2",
                "subject2",
                "subject3",
                "subject3",
            ],
            "time": pd.date_range(start="2020-01-01", periods=6, freq="5min"),
            "gl": [150, 200, 130, 190, 140, 140],
        }
    )

    # Calculate mean_glu
    result = mean_glu(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "mean" in result.columns
    assert len(result) == 3  # Three subjects

    # Check that mean values are correct
    subject1_mean = result[result["id"] == "subject1"]["mean"].iloc[0]
    subject2_mean = result[result["id"] == "subject2"]["mean"].iloc[0]
    subject3_mean = result[result["id"] == "subject3"]["mean"].iloc[0]
    assert abs(subject1_mean - 175.0) < 0.01  # 175.0 is the expected mean
    assert abs(subject2_mean - 160.0) < 0.01  # 160.0 is the expected mean
    assert abs(subject3_mean - 140.0) < 0.01  # 140.0 is the expected mean
