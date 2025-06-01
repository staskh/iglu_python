import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu
from iglu_python.grade import _grade_formula, grade

method_name = "grade"


def get_test_scenarios():
    """Get test scenarios for GRADE calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for GRADE method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_grade_iglu_r_compatible(scenario):
    """Test GRADE calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.grade(df, **kwargs)

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


def test_grade_formula():
    """Test the helper function that calculates GRADE scores for individual values."""
    # Test with perfect glucose value (should give low GRADE score)
    assert _grade_formula(np.array([100])) < 10

    # Test with high glucose value (should give high GRADE score)
    assert _grade_formula(np.array([300])) > 20

    # Test with very high glucose value (should be capped at 50)
    assert _grade_formula(np.array([1000])) <= 50

    # Test with multiple values
    values = np.array([100, 200, 300])
    scores = _grade_formula(values)
    assert len(scores) == 3
    assert all(scores >= 0) and all(scores <= 50)


def test_grade_basic():
    """Test basic GRADE calculation with known glucose values."""
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

    result = grade(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "GRADE" in result.columns
    assert len(result) == 2

    # Check calculations
    # Subject 1 has higher GRADE score due to higher glucose values
    assert (
        result.loc[result["id"] == "subject1", "GRADE"].values[0]
        > result.loc[result["id"] == "subject2", "GRADE"].values[0]
    )


def test_grade_series_input():
    """Test GRADE calculation with Series input."""
    data = pd.Series([100, 200, 100, 100])
    result = grade(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "GRADE" in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1


def test_grade_empty_data():
    """Test GRADE calculation with empty DataFrame."""
    data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        grade(data)


def test_grade_missing_values():
    """Test GRADE calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [100, np.nan, 200],
        }
    )

    result = grade(data)

    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, "GRADE"])
    assert len(result) == 1


def test_grade_constant_values():
    """Test GRADE calculation with constant glucose values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [100, 100, 100],  # All values are the same
        }
    )

    result = grade(data)

    # Check that GRADE score is consistent
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "GRADE"] == _grade_formula(np.array([100]))[0]


def test_grade_multiple_subjects():
    """Test GRADE calculation with multiple subjects."""
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
            "gl": [100, 100, 200, 200, 100, 200],  # Different patterns for each subject
        }
    )

    result = grade(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result["id"]) == {"subject1", "subject2", "subject3"}

    # Check relative values
    # Subject 1 has best control (lowest GRADE)
    assert (
        result.loc[result["id"] == "subject1", "GRADE"].values[0]
        < result.loc[result["id"] == "subject2", "GRADE"].values[0]
    )
    # Subject 2 has worst control (highest GRADE)
    assert (
        result.loc[result["id"] == "subject2", "GRADE"].values[0]
        > result.loc[result["id"] == "subject3", "GRADE"].values[0]
    )
    # Subject 3 has mixed control (middle GRADE)
    assert (
        result.loc[result["id"] == "subject3", "GRADE"].values[0]
        > result.loc[result["id"] == "subject1", "GRADE"].values[0]
    )
