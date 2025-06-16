import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "grade_hypo"


def get_test_scenarios():
    """Get test scenarios for GRADE hypoglycemia calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for GRADE hypoglycemia method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_grade_hypo_iglu_r_compatible(scenario):
    """Test GRADE hypoglycemia calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.grade_hypo(df, **kwargs)

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


def test_grade_hypo_default():
    """Test GRADE hypoglycemia with default parameters"""
    data = pd.DataFrame(
        {
            "id": [
                "subject1",
                "subject1",
                "subject1",
                "subject1",
                "subject2",
                "subject2",
            ],
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:10:00",
                    "2020-01-01 00:15:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                ]
            ),
            "gl": [150, 75, 160, 65, 140, 85],  # Include some hypoglycemic values
        }
    )

    result = iglu.grade_hypo(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "GRADE_hypo"])
    assert len(result) == 2  # One row per subject
    assert all(result["GRADE_hypo"] >= 0)  # Percentages should be non-negative
    assert all(result["GRADE_hypo"] <= 100)  # Percentages should not exceed 100%


def test_grade_hypo_series_input():
    """Test GRADE hypoglycemia with Series input"""
    series_data = pd.Series(
        [150, 75, 160, 65, 140, 85]
    )  # Include some hypoglycemic values
    result = iglu.grade_hypo(series_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 19.225537, rtol=1e-3)

def test_grade_hypo_list_input():
    """Test GRADE hypoglycemia with Series input"""
    list_data = [150, 75, 160, 65, 140, 85]
    result = iglu.grade_hypo(list_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 19.225537, rtol=1e-3)

def test_grade_hypo_numpy_array_input():
    """Test GRADE hypoglycemia with Series input"""
    array_data = np.array([150, 75, 160, 65, 140, 85])
    result = iglu.grade_hypo(array_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 19.225537, rtol=1e-3)

def test_grade_hypo_empty():
    """Test GRADE hypoglycemia with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.grade_hypo(empty_data)


def test_grade_hypo_constant_glucose():
    """Test GRADE hypoglycemia with constant glucose values"""
    # Test with constant glucose above lower bound
    series_data = pd.Series([150] * 10)
    result = iglu.grade_hypo(series_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 0, rtol=1e-3)

    # Test with constant glucose below lower bound
    series_data = pd.Series([70] * 10)
    result = iglu.grade_hypo(series_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 100, rtol=1e-3)


def test_grade_hypo_missing_values():
    """Test GRADE hypoglycemia with missing values"""
    data_with_na = pd.DataFrame(
        {
            "id": ["subject1"] * 4,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:10:00",
                    "2020-01-01 00:15:00",
                ]
            ),
            "gl": [150, np.nan, 75, 65],
        }
    )
    result = iglu.grade_hypo(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["GRADE_hypo"].iloc[0] >= 0
    assert result["GRADE_hypo"].iloc[0] <= 100


def test_grade_hypo_different_lower():
    """Test GRADE hypoglycemia with different lower bounds"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 6,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:10:00",
                    "2020-01-01 00:15:00",
                    "2020-01-01 00:20:00",
                    "2020-01-01 00:25:00",
                ]
            ),
            "gl": [150, 75, 160, 65, 140, 85],
        }
    )

    result_80 = iglu.grade_hypo(data, lower=80)
    result_70 = iglu.grade_hypo(data, lower=70)
    assert len(result_80) == 1
    assert len(result_70) == 1
    assert (
        result_70["GRADE_hypo"].iloc[0] != result_80["GRADE_hypo"].iloc[0]
    )  # Different lower bounds should give different results


def test_grade_hypo_extreme_values():
    """Test GRADE hypoglycemia with extreme glucose values"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 4,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:10:00",
                    "2020-01-01 00:15:00",
                ]
            ),
            "gl": [
                40,
                400,
                30,
                500,
            ],  # Extreme values both below and above normal range
        }
    )

    result = iglu.grade_hypo(data)
    assert len(result) == 1
    assert result["GRADE_hypo"].iloc[0] >= 0
    assert result["GRADE_hypo"].iloc[0] <= 100


def test_grade_hypo_basic():
    """Test basic GRADE hypoglycemia calculation with known glucose values."""
    # ... existing code ...
