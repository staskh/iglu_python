import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu
from iglu_python.grade_eugly import grade_eugly

method_name = "grade_eugly"


def get_test_scenarios():
    """Get test scenarios for GRADE euglycemia calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for GRADE euglycemia method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_grade_eugly_iglu_r_compatible(scenario):
    """Test GRADE euglycemia calculation against expected results"""

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

    result_df = iglu.grade_eugly(df, **kwargs)

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


def test_grade_eugly_basic():
    """Test basic GRADE euglycemia calculation with known glucose values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject2", "subject2"],
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                ]
            ),
            "gl": [100, 200, 100, 100],  # One subject has better control
        }
    )

    result = grade_eugly(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "GRADE_eugly" in result.columns
    assert len(result) == 2

    # Check calculations
    # Subject 1 has lower euglycemia percentage due to high glucose value
    assert (
        result.loc[result["id"] == "subject1", "GRADE_eugly"].values[0]
        < result.loc[result["id"] == "subject2", "GRADE_eugly"].values[0]
    )


def test_grade_eugly_series_input():
    """Test GRADE euglycemia calculation with Series input."""
    data = pd.Series([100, 200, 100, 100])
    result = grade_eugly(data)

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 8.70927656, rtol=1e-3)


def test_grade_eugly_custom_targets():
    """Test GRADE euglycemia calculation with custom target ranges."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1"],
            "time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:05:00"]),
            "gl": [100, 200],
        }
    )

    # Test with different target ranges
    result1 = grade_eugly(data, lower=70, upper=140)
    result2 = grade_eugly(data, lower=70, upper=180)

    # More values should be in range with wider targets
    assert result1.loc[0, "GRADE_eugly"] <= result2.loc[0, "GRADE_eugly"]


def test_grade_eugly_empty_data():
    """Test GRADE euglycemia calculation with empty DataFrame."""
    data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        grade_eugly(data)


def test_grade_eugly_missing_values():
    """Test GRADE euglycemia calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [100, np.nan, 200],
        }
    )

    result = grade_eugly(data)

    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, "GRADE_eugly"])
    assert len(result) == 1


def test_grade_eugly_all_out_of_range():
    """Test GRADE euglycemia calculation when all values are out of range."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [50, 300, 400],  # All values outside default range (70-140)
        }
    )

    result = grade_eugly(data)

    # Check that euglycemia percentage is 0
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "GRADE_eugly"] == 0


def test_grade_eugly_all_in_range():
    """Test GRADE euglycemia calculation when all values are in range."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [80, 100, 120],  # All values within default range (70-140)
        }
    )

    result = grade_eugly(data)

    # Check that euglycemia percentage is 100
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "GRADE_eugly"] == 100


def test_grade_eugly_multiple_subjects():
    """Test GRADE euglycemia calculation with multiple subjects."""
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
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                ]
            ),
            "gl": [80, 80, 200, 200, 80, 200],  # Different patterns for each subject
        }
    )

    result = grade_eugly(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result["id"]) == {"subject1", "subject2", "subject3"}

    # Check relative values
    # Subject 1 has best control (highest euglycemia)
    assert (
        result.loc[result["id"] == "subject1", "GRADE_eugly"].values[0]
        > result.loc[result["id"] == "subject2", "GRADE_eugly"].values[0]
    )
    # Subject 2 has worst control (lowest euglycemia)
    assert (
        result.loc[result["id"] == "subject2", "GRADE_eugly"].values[0]
        < result.loc[result["id"] == "subject3", "GRADE_eugly"].values[0]
    )
    # Subject 3 has mixed control (middle euglycemia)
    assert (
        result.loc[result["id"] == "subject3", "GRADE_eugly"].values[0]
        < result.loc[result["id"] == "subject1", "GRADE_eugly"].values[0]
    )


def test_grade_eugly_list_input():
    """Test GRADE euglycemia calculation with list input."""
    # Create test data as a list
    data = [70, 80, 90, 100, 110, 120, 130, 140, 150, 160]  # 4 values in euglycemic range (70-140)

    # Calculate GRADE euglycemia
    result = iglu.grade_eugly(data)

    # Expected results:
    # Euglycemic values: 80, 90, 100, 110, 120, 130, 140 (7 values)
    # Total values: 10
    # GRADE euglycemia = (7 / 10) * 100 = 70.0
    expected = 51.496907

    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)


def test_grade_eugly_numpy_array_input():
    """Test GRADE euglycemia calculation with numpy array input."""
    # Create test data as a numpy array
    data = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160])  # 4 values in euglycemic range (70-140)

    # Calculate GRADE euglycemia
    result = iglu.grade_eugly(data)

    # Expected results:
    # Euglycemic values: 80, 90, 100, 110, 120, 130, 140 (7 values)
    # Total values: 10
    # GRADE euglycemia = (7 / 10) * 100 = 70.0
    expected = 51.496907

    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)


def test_grade_eugly_list_with_missing_values():
    """Test GRADE euglycemia calculation with list input containing missing values."""
    # Create test data as a list with None values
    data = [70, None, 90, 100, None, 120, 130, 140, None, 160]  # 5 values in euglycemic range (70-140)

    # Calculate GRADE euglycemia
    result = iglu.grade_eugly(data)

    # Expected results:
    # Euglycemic values: 90, 100, 120, 130, 140 (5 values)
    # Total non-None values: 7
    # GRADE euglycemia = (5 / 7) * 100 = 71.429
    expected = 63.297202

    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)


def test_grade_eugly_numpy_array_with_nan():
    """Test GRADE euglycemia calculation with numpy array input containing NaN values."""
    # Create test data as a numpy array with NaN values
    data = np.array([70, np.nan, 90, 100, np.nan, 120, 130, 140, np.nan, 160])  # 5 values in euglycemic range (70-140)

    # Calculate GRADE euglycemia
    result = iglu.grade_eugly(data)

    # Expected results:
    # Euglycemic values: 90, 100, 120, 130, 140 (5 values)
    # Total non-NaN values: 7
    # GRADE euglycemia = (5 / 7) * 100 = 71.429
    expected = 63.297202

    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)


def test_grade_eugly_empty_input():
    """Test GRADE euglycemia calculation with empty input."""
    # Test with empty list
    result = iglu.grade_eugly([])
    assert isinstance(result, float)
    assert np.isnan(result)

    # Test with empty numpy array
    result = iglu.grade_eugly(np.array([]))
    assert isinstance(result, float)
    assert np.isnan(result)


def test_grade_eugly_boundary_values():
    """Test GRADE euglycemia calculation with boundary values."""
    # Create test data with boundary values (70 and 140)
    data = [60, 70, 80, 140, 150]  # 70 and 140 are included in euglycemic range

    # Calculate GRADE euglycemia
    result = iglu.grade_eugly(data)

    # Expected results:
    # Euglycemic values: 70, 80, 140 (3 values)
    # Total values: 5
    # GRADE euglycemia = (3 / 5) * 100 = 60.0
    expected = 36.910589

    # Compare results
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=0.001)
