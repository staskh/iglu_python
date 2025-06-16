import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "igc"


def get_test_scenarios():
    """Get test scenarios for IGC calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for IGC method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_igc_iglu_r_compatible(scenario):
    """Test IGC calculation against expected results"""

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

    result_df = iglu.igc(df, **kwargs)

    assert result_df is not None

    # Compare DataFrames with precision to 0.001 for numeric columns
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


def test_igc_basic():
    """Test basic IGC calculation with known values."""
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
            "gl": [80, 100, 120, 130, 140, 160],
        }
    )

    # Calculate IGC
    result = iglu.igc(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "IGC" in result.columns
    assert len(result) == 2  # Two subjects

    # Check that IGC values are non-negative
    assert all(result["IGC"] >= 0)

    # Check that IGC values are reasonable (should be between 0 and 1 for this data)
    assert all(result["IGC"] <= 1)


def test_igc_series_input():
    """Test IGC calculation with Series input."""
    # Create test data as Series
    data = pd.Series([150, 200, 180, 130, 190, 160])

    # Calculate IGC
    result = iglu.igc(data)
    expected = 1.453976

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_igc_list_input():
    """Test IGC calculation with list input."""
    data = [150, 200, 180, 130, 190, 160]
    result = iglu.igc(data)
    expected = 1.453976
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_igc_numpy_array_input():
    """Test IGC calculation with numpy array input."""
    data = np.array([150, 200, 180, 130, 190, 160])
    result = iglu.igc(data)
    expected = 1.453976
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)



def test_igc_custom_parameters():
    """Test IGC calculation with custom parameters."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, 200, 180],
        }
    )

    # Test with custom parameters
    result = iglu.igc(data, LLTR=70, ULTR=180, a=1.5, b=1.8, c=25, d=25)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "IGC" in result.columns
    assert len(result) == 1

    # Check that IGC value is non-negative
    assert result["IGC"].iloc[0] >= 0


def test_igc_empty_data():
    """Test IGC calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=["id", "time", "gl"])

    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        iglu.igc(data)


def test_igc_missing_values():
    """Test IGC calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, np.nan, 180],
        }
    )

    # Calculate IGC
    result = iglu.igc(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "IGC" in result.columns
    assert len(result) == 1

    # Check that IGC value is non-negative
    assert result["IGC"].iloc[0] >= 0


def test_igc_constant_glucose():
    """Test IGC calculation with constant glucose values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [140, 140, 140],  # Constant value at ULTR
        }
    )

    # Calculate IGC
    result = iglu.igc(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "IGC" in result.columns
    assert len(result) == 1

    # Check that IGC value is non-negative
    assert result["IGC"].iloc[0] >= 0


def test_igc_extreme_values():
    """Test IGC calculation with extreme glucose values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [40, 400, 140],  # Very low, very high, and target value
        }
    )

    # Calculate IGC
    result = iglu.igc(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "IGC" in result.columns
    assert len(result) == 1

    # Check that IGC value is non-negative
    assert result["IGC"].iloc[0] >= 0

    # Check that IGC value is higher than for normal values
    # (since we have extreme values)
    normal_data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [140, 140, 140],
        }
    )
    normal_result = iglu.igc(normal_data)
    assert result["IGC"].iloc[0] > normal_result["IGC"].iloc[0]


def test_igc_multiple_subjects():
    """Test IGC calculation with multiple subjects."""
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

    # Calculate IGC
    result = iglu.igc(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "IGC" in result.columns
    assert len(result) == 3  # Three subjects

    # Check that IGC values are non-negative
    assert all(result["IGC"] >= 0)

    # Check that subject3 has lower IGC than others (since values are at target)
    subject3_igc = result[result["id"] == "subject3"]["IGC"].iloc[0]
    subject1_igc = result[result["id"] == "subject1"]["IGC"].iloc[0]
    subject2_igc = result[result["id"] == "subject2"]["IGC"].iloc[0]
    assert subject3_igc <= subject1_igc
    assert subject3_igc <= subject2_igc
