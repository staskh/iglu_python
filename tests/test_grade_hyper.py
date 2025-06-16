import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu
from iglu_python.grade_hyper import grade_hyper

method_name = "grade_hyper"


def get_test_scenarios():
    """Get test scenarios for GRADE hyperglycemia calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for GRADE hyperglycemia method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_grade_hyper_iglu_r_compatible(scenario):
    """Test GRADE hyperglycemia calculation against expected results"""

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

    result_df = iglu.grade_hyper(df, **kwargs)

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


def test_grade_hyper_basic():
    """Test basic GRADE hyperglycemia calculation with known glucose values."""
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
            "gl": [150, 200, 130, 190],  # One subject has more hyperglycemia
        }
    )

    result = grade_hyper(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "GRADE_hyper" in result.columns
    assert len(result) == 2

    # Check calculations
    # Subject 1 has higher hyperglycemia percentage due to higher glucose values
    assert (
        result.loc[result["id"] == "subject1", "GRADE_hyper"].values[0]
        > result.loc[result["id"] == "subject2", "GRADE_hyper"].values[0]
    )


def test_grade_hyper_series_input():
    """Test GRADE hyperglycemia calculation with Series input."""
    data = pd.Series([150, 200, 130, 190])
    result = grade_hyper(data)

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 89.684900, rtol=1e-3)

def test_grade_hyper_list_input():
    """Test GRADE hyperglycemia calculation with Series input."""
    data = [150, 200, 130, 190]
    result = grade_hyper(data)

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 89.684900, rtol=1e-3)

def test_grade_hyper_numpy_array_input():
    """Test GRADE hyperglycemia calculation with Series input."""
    data = np.array([150, 200, 130, 190])
    result = grade_hyper(data)

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 89.684900, rtol=1e-3)

def test_grade_hyper_custom_upper():
    """Test GRADE hyperglycemia calculation with custom upper bound."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1"],
            "time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:05:00"]),
            "gl": [150, 200],
        }
    )

    # Test with different upper bounds
    result1 = grade_hyper(data, upper=140)
    result2 = grade_hyper(data, upper=180)

    # More values should be hyperglycemic with lower upper bound
    assert result1.loc[0, "GRADE_hyper"] > result2.loc[0, "GRADE_hyper"]


def test_grade_hyper_empty_data():
    """Test GRADE hyperglycemia calculation with empty DataFrame."""
    data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        grade_hyper(data)


def test_grade_hyper_missing_values():
    """Test GRADE hyperglycemia calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [150, np.nan, 200],
        }
    )

    result = grade_hyper(data)

    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, "GRADE_hyper"])
    assert len(result) == 1


def test_grade_hyper_all_below_upper():
    """Test GRADE hyperglycemia calculation when all values are below upper bound."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [80, 100, 120],  # All values below default upper bound (140)
        }
    )

    result = grade_hyper(data)

    # Check that hyperglycemia percentage is 0
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "GRADE_hyper"] == 0


def test_grade_hyper_all_above_upper():
    """Test GRADE hyperglycemia calculation when all values are above upper bound."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [200, 250, 300],  # All values above default upper bound (140)
        }
    )

    result = grade_hyper(data)

    # Check that hyperglycemia percentage is 100
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "GRADE_hyper"] == 100


def test_grade_hyper_multiple_subjects():
    """Test GRADE hyperglycemia calculation with multiple subjects."""
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

    result = grade_hyper(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result["id"]) == {"subject1", "subject2", "subject3"}

    # Check relative values
    # Subject 1 has best control (lowest hyperglycemia)
    assert (
        result.loc[result["id"] == "subject1", "GRADE_hyper"].values[0]
        < result.loc[result["id"] == "subject2", "GRADE_hyper"].values[0]
    )
    # Subject 2 has worst control (highest hyperglycemia)
    assert (
        result.loc[result["id"] == "subject2", "GRADE_hyper"].values[0]
        > result.loc[result["id"] == "subject3", "GRADE_hyper"].values[0]
    )
    # Subject 3 has mixed control (middle hyperglycemia)
    assert (
        result.loc[result["id"] == "subject3", "GRADE_hyper"].values[0]
        > result.loc[result["id"] == "subject1", "GRADE_hyper"].values[0]
    )
