import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "gri"


def get_test_scenarios():
    """Get test scenarios for GRI calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for GRI method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_gri_iglu_r_compatible(scenario):
    """Test GRI calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.gri(df, **kwargs)

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


def test_gri_default():
    """Test GRI with default parameters"""
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
            "gl": [150, 50, 160, 260, 140, 85],  # Include values in all GRI ranges
        }
    )

    result = iglu.gri(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "GRI"])
    assert len(result) == 2  # One row per subject
    assert all(result["GRI"] >= 0)  # GRI should be non-negative
    assert all(result["GRI"] <= 100)  # GRI should not exceed 100%


def test_gri_series():
    """Test GRI with Series input"""
    series_data = pd.Series(
        [150, 50, 160, 260, 140, 85]
    )  # Include values in all GRI ranges
    result = iglu.gri(series_data)
    assert isinstance(result, pd.DataFrame)
    assert "GRI" in result.columns
    assert len(result) == 1
    assert result["GRI"].iloc[0] >= 0
    assert result["GRI"].iloc[0] <= 100


def test_gri_empty():
    """Test GRI with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.gri(empty_data)


def test_gri_constant_glucose():
    """Test GRI with constant glucose values"""
    # Test with constant glucose in target range
    series_data = pd.Series([150] * 10)
    result = iglu.gri(series_data)
    assert len(result) == 1
    assert (
        result["GRI"].iloc[0] == 0
    )  # Should be 0 for constant glucose in target range

    # Test with constant glucose in severe hypoglycemia range
    series_data = pd.Series([40] * 10)
    result = iglu.gri(series_data)
    assert len(result) == 1
    assert result["GRI"].iloc[0] > 0  # Should be positive for constant glucose below 54


def test_gri_missing_values():
    """Test GRI with missing values"""
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
            "gl": [150, np.nan, 50, 260],
        }
    )
    result = iglu.gri(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["GRI"].iloc[0] >= 0
    assert result["GRI"].iloc[0] <= 100


def test_gri_extreme_values():
    """Test GRI with extreme glucose values"""
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

    result = iglu.gri(data)
    assert len(result) == 1
    assert result["GRI"].iloc[0] >= 0
    assert result["GRI"].iloc[0] <= 100


def test_gri_timezone():
    """Test GRI with timezone parameter"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 8,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:10:00",
                    "2020-01-01 00:15:00",
                    "2020-01-01 00:20:00",
                    "2020-01-01 00:25:00",
                    "2020-01-01 00:30:00",
                    "2020-01-01 00:35:00",
                ]
            ),
            "gl": [150, 50, 160, 260, 140, 85, 200, 45],
        }
    )

    result = iglu.gri(data, tz="UTC")
    assert len(result) == 1
    assert isinstance(result["GRI"].iloc[0], (np.int64, float))
    assert result["GRI"].iloc[0] >= 0
    assert result["GRI"].iloc[0] <= 100
