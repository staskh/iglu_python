import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "in_range_percent"


def get_test_scenarios():
    """Get test scenarios for in_range_percent calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for in_range_percent method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_in_range_percent_iglu_r_compatible(scenario):
    """Test in_range_percent calculation against expected results"""

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

    result_df = iglu.in_range_percent(df, **kwargs)

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


def test_in_range_percent_basic():
    """Test basic in_range_percent calculation with known values."""
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
            "gl": [80, 90, 100, 130, 190, 160],
        }
    )

    # Calculate in_range_percent
    result = iglu.in_range_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "in_range_70_180" in result.columns
    assert "in_range_63_140" in result.columns
    assert len(result) == 2  # Two subjects

    # Check that percentages are between 0 and 100
    assert all((result["in_range_70_180"] >= 0) & (result["in_range_70_180"] <= 100))
    assert all((result["in_range_63_140"] >= 0) & (result["in_range_63_140"] <= 100))

    # Check that subject1 has higher percentages than subject2
    # (since subject1 has more values in range)
    subject1_in_range = result[result["id"] == "subject1"]["in_range_70_180"].iloc[0]
    subject2_in_range = result[result["id"] == "subject2"]["in_range_70_180"].iloc[0]
    assert subject1_in_range > subject2_in_range


def test_in_range_percent_series_input():
    """Test in_range_percent calculation with Series input."""
    # Create test data as Series
    data = pd.Series([80, 90, 100, 130, 190, 160])

    # Calculate in_range_percent
    result = iglu.in_range_percent(data)

    # Check output format
    assert isinstance(result, dict)
    assert "in_range_70_180" in result
    assert "in_range_63_140" in result
    assert len(result) == 2

    # Check that percentages are between 0 and 100
    np.testing.assert_allclose(result["in_range_70_180"], 83.33, rtol=1e-3)
    np.testing.assert_allclose(result["in_range_63_140"], 66.66, rtol=1e-3)

def test_in_range_percent_list_input():
    """Test in_range_percent calculation with list input."""
    data = [80, 90, 100, 130, 190, 160]
    result = iglu.in_range_percent(data)
    assert isinstance(result, dict)
    assert "in_range_70_180" in result
    assert "in_range_63_140" in result
    assert len(result) == 2

    # Check that percentages are between 0 and 100
    np.testing.assert_allclose(result["in_range_70_180"], 83.33, rtol=1e-3)
    np.testing.assert_allclose(result["in_range_63_140"], 66.66, rtol=1e-3)

def test_in_range_percent_numpy_array_input():
    """Test in_range_percent calculation with numpy array input."""
    data = np.array([80, 90, 100, 130, 190, 160])
    result = iglu.in_range_percent(data)
    assert isinstance(result, dict)
    assert "in_range_70_180" in result
    assert "in_range_63_140" in result
    assert len(result) == 2

    # Check that percentages are between 0 and 100
    np.testing.assert_allclose(result["in_range_70_180"], 83.33, rtol=1e-3)
    np.testing.assert_allclose(result["in_range_63_140"], 66.66, rtol=1e-3)

def test_in_range_percent_custom_targets():
    """Test in_range_percent calculation with custom targets."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [80, 90, 100],
        }
    )

    # Test with custom targets
    result = iglu.in_range_percent(data, target_ranges=[[75, 95], [85, 105]])

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "in_range_75_95" in result.columns
    assert "in_range_85_105" in result.columns
    assert len(result) == 1

    # Check that percentages are between 0 and 100
    assert (result["in_range_75_95"].iloc[0] >= 0) and (
        result["in_range_75_95"].iloc[0] <= 100
    )
    assert (result["in_range_85_105"].iloc[0] >= 0) and (
        result["in_range_85_105"].iloc[0] <= 100
    )


def test_in_range_percent_empty_data():
    """Test in_range_percent calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=["id", "time", "gl"])

    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        iglu.in_range_percent(data)


def test_in_range_percent_missing_values():
    """Test in_range_percent calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [80, np.nan, 100],
        }
    )

    # Calculate in_range_percent
    result = iglu.in_range_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "in_range_70_180" in result.columns
    assert "in_range_63_140" in result.columns
    assert len(result) == 1

    # Check that percentages are between 0 and 100
    assert (result["in_range_70_180"].iloc[0] >= 0) and (
        result["in_range_70_180"].iloc[0] <= 100
    )
    assert (result["in_range_63_140"].iloc[0] >= 0) and (
        result["in_range_63_140"].iloc[0] <= 100
    )


def test_in_range_percent_all_out_of_range():
    """Test in_range_percent calculation with all values out of range."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [50, 60, 200],  # All values outside [70, 180]
        }
    )

    # Calculate in_range_percent
    result = iglu.in_range_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "in_range_70_180" in result.columns
    assert "in_range_63_140" in result.columns
    assert len(result) == 1

    # Check that percentages are 0 for all ranges
    assert result["in_range_70_180"].iloc[0] == 0
    assert result["in_range_63_140"].iloc[0] == 0


def test_in_range_percent_all_in_range():
    """Test in_range_percent calculation with all values in range."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [80, 90, 100],  # All values within [70, 180]
        }
    )

    # Calculate in_range_percent
    result = iglu.in_range_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "in_range_70_180" in result.columns
    assert "in_range_63_140" in result.columns
    assert len(result) == 1

    # Check that percentages are 100 for all ranges
    assert result["in_range_70_180"].iloc[0] == 100
    assert result["in_range_63_140"].iloc[0] == 100


def test_in_range_percent_multiple_subjects():
    """Test in_range_percent calculation with multiple subjects."""
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
            "gl": [80, 90, 130, 190, 140, 190],
        }
    )

    # Calculate in_range_percent
    result = iglu.in_range_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "in_range_70_180" in result.columns
    assert "in_range_63_140" in result.columns
    assert len(result) == 3  # Three subjects

    # Check that percentages are between 0 and 100
    assert all((result["in_range_70_180"] >= 0) & (result["in_range_70_180"] <= 100))
    assert all((result["in_range_63_140"] >= 0) & (result["in_range_63_140"] <= 100))

    # Check that subject1 has higher percentages than others
    subject1_in_range = result[result["id"] == "subject1"]["in_range_70_180"].iloc[0]
    subject2_in_range = result[result["id"] == "subject2"]["in_range_70_180"].iloc[0]
    subject3_in_range = result[result["id"] == "subject3"]["in_range_70_180"].iloc[0]
    assert subject1_in_range > subject2_in_range
    assert subject1_in_range > subject3_in_range
