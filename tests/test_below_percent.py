import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "below_percent"


def get_test_scenarios():
    """Get test scenarios for above_percent calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for above_percent method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.fixture
def test_data():
    """Fixture that provides test data for above_percent calculations"""
    return get_test_scenarios()


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_below_percent_iglu_r_compatible(scenario):
    """Test below_percent calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.below_percent(df, **kwargs)

    assert result_df is not None

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Compare DataFrames with precision to 0.001
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


def test_below_percent_basic():
    """Test basic below_percent calculation with known values."""
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
            "gl": [50, 60, 65, 130, 190, 160],
        }
    )

    # Calculate below_percent
    result = iglu.below_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "below_54" in result.columns
    assert "below_70" in result.columns
    assert len(result) == 2  # Two subjects

    # Check that percentages are between 0 and 100
    assert all((result["below_54"] >= 0) & (result["below_54"] <= 100))
    assert all((result["below_70"] >= 0) & (result["below_70"] <= 100))

    # Check that subject1 has higher percentages than subject2
    # (since subject1 has values below thresholds)
    subject1_below_54 = result[result["id"] == "subject1"]["below_54"].iloc[0]
    subject2_below_54 = result[result["id"] == "subject2"]["below_54"].iloc[0]
    assert subject1_below_54 > subject2_below_54


def test_below_percent_series_input():
    """Test below_percent calculation with Series input."""
    # Create test data as Series
    data = pd.Series([50, 60, 65, 130, 190, 160])

    # Calculate below_percent
    result = iglu.below_percent(data)

    # Check output format
    assert isinstance(result, dict)
    assert "below_54" in result
    assert "below_70" in result
    assert len(result) == 2

    # Check that percentages are between 0 and 100
    assert (result["below_54"] >= 0) and (result["below_54"] <= 100)
    assert (result["below_70"] >= 0) and (result["below_70"] <= 100)


def test_below_percent_custom_targets():
    """Test below_percent calculation with custom targets."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [50, 60, 65],
        }
    )

    # Test with custom targets
    result = iglu.below_percent(data, targets_below=[55, 65])

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "below_55" in result.columns
    assert "below_65" in result.columns
    assert len(result) == 1

    # Check that percentages are between 0 and 100
    assert (result["below_55"].iloc[0] >= 0) and (result["below_55"].iloc[0] <= 100)
    assert (result["below_65"].iloc[0] >= 0) and (result["below_65"].iloc[0] <= 100)


def test_below_percent_empty_data():
    """Test below_percent calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=["id", "time", "gl"])

    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        iglu.below_percent(data)


def test_below_percent_missing_values():
    """Test below_percent calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [50, np.nan, 65],
        }
    )

    # Calculate below_percent
    result = iglu.below_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "below_54" in result.columns
    assert "below_70" in result.columns
    assert len(result) == 1

    # Check that percentages are between 0 and 100
    assert (result["below_54"].iloc[0] >= 0) and (result["below_54"].iloc[0] <= 100)
    assert (result["below_70"].iloc[0] >= 0) and (result["below_70"].iloc[0] <= 100)


def test_below_percent_all_above():
    """Test below_percent calculation with all values above thresholds."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, 160, 170],  # All values above 70
        }
    )

    # Calculate below_percent
    result = iglu.below_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "below_54" in result.columns
    assert "below_70" in result.columns
    assert len(result) == 1

    # Check that percentages are 0 for all thresholds
    assert result["below_54"].iloc[0] == 0
    assert result["below_70"].iloc[0] == 0


def test_below_percent_all_below():
    """Test below_percent calculation with all values below thresholds."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [40, 45, 50],  # All values below 54
        }
    )

    # Calculate below_percent
    result = iglu.below_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "below_54" in result.columns
    assert "below_70" in result.columns
    assert len(result) == 1

    # Check that percentages are 100 for all thresholds
    assert result["below_54"].iloc[0] == 100
    assert result["below_70"].iloc[0] == 100


def test_below_percent_multiple_subjects():
    """Test below_percent calculation with multiple subjects."""
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
            "gl": [50, 60, 130, 190, 140, 140],
        }
    )

    # Calculate below_percent
    result = iglu.below_percent(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "below_54" in result.columns
    assert "below_70" in result.columns
    assert len(result) == 3  # Three subjects

    # Check that percentages are between 0 and 100
    assert all((result["below_54"] >= 0) & (result["below_54"] <= 100))
    assert all((result["below_70"] >= 0) & (result["below_70"] <= 100))

    # Check that subject1 has higher percentages than others
    subject1_below_70 = result[result["id"] == "subject1"]["below_70"].iloc[0]
    subject2_below_70 = result[result["id"] == "subject2"]["below_70"].iloc[0]
    subject3_below_70 = result[result["id"] == "subject3"]["below_70"].iloc[0]
    assert subject1_below_70 > subject2_below_70
    assert subject1_below_70 > subject3_below_70
