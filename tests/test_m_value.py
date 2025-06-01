import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "m_value"


def get_test_scenarios():
    """Get test scenarios for M-value calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for M-value method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_m_value_iglu_r_compatible(scenario):
    """Test M-value calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.m_value(df, **kwargs)

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


def test_m_value_basic():
    """Test basic M-value calculation with known glucose values."""
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
            "gl": [
                90,
                180,
                90,
                90,
            ],  # One subject has perfect control, other has high values
        }
    )

    result = iglu.m_value(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "M_value" in result.columns
    assert len(result) == 2

    # Check calculations
    # Subject 2 has perfect control (all values at reference), should have M-value close to 0
    assert result.loc[result["id"] == "subject2", "M_value"].values[0] < 1
    # Subject 1 has high values, should have higher M-value
    assert result.loc[result["id"] == "subject1", "M_value"].values[0] > 10


def test_m_value_series_input():
    """Test M-value calculation with Series input."""
    data = pd.Series([90, 180, 90, 90])
    result = iglu.m_value(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "M_value" in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1


def test_m_value_custom_reference():
    """Test M-value calculation with custom reference value."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1"],
            "time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:05:00"]),
            "gl": [100, 200],
        }
    )

    # Test with different reference values
    result1 = iglu.m_value(data, r=100)
    result2 = iglu.m_value(data, r=150)

    # M-value should be lower with higher reference value for these data
    assert result1.loc[0, "M_value"] > result2.loc[0, "M_value"]


def test_m_value_empty_data():
    """Test M-value calculation with empty DataFrame."""
    data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.m_value(data)


def test_m_value_missing_values():
    """Test M-value calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [90, np.nan, 180],
        }
    )

    result = iglu.m_value(data)

    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, "M_value"])
    assert len(result) == 1


def test_m_value_constant_values():
    """Test M-value calculation with constant glucose values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [90, 90, 90],  # All values at reference
        }
    )

    result = iglu.m_value(data)

    # M-value should be very close to 0 for perfect control
    assert result.loc[0, "M_value"] < 1


def test_m_value_multiple_subjects():
    """Test M-value calculation with multiple subjects."""
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
            "gl": [90, 90, 180, 180, 90, 180],  # Different patterns for each subject
        }
    )

    result = iglu.m_value(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result["id"]) == {"subject1", "subject2", "subject3"}

    # Check relative values
    # Subject 1 has perfect control
    assert result.loc[result["id"] == "subject1", "M_value"].values[0] < 1
    # Subject 2 has high values
    assert result.loc[result["id"] == "subject2", "M_value"].values[0] > 10
    # Subject 3 has mixed values
    m_value3 = result.loc[result["id"] == "subject3", "M_value"].values[0]
    assert 1 < m_value3 < 20
