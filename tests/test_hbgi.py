import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "hbgi"


def get_test_scenarios():
    """Get test scenarios for HBGI calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])

    # Filter scenarios for HBGI method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_hbgi_iglu_r_compatible(scenario):
    """Test HBGI calculation against expected results"""

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

    result_df = iglu.hbgi(df, **kwargs)

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


def test_hbgi_basic():
    """Test basic HBGI calculation with known glucose values."""
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
            "gl": [150, 200, 130, 190],  # Different hyperglycemia for each subject
        }
    )

    result = iglu.hbgi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "HBGI" in result.columns
    assert len(result) == 2

    # Check calculations
    # Subject 1 has higher hyperglycemia (higher glucose values)
    assert (
        result.loc[result["id"] == "subject1", "HBGI"].values[0]
        > result.loc[result["id"] == "subject2", "HBGI"].values[0]
    )


def test_hbgi_series_input():
    """Test HBGI calculation with Series input."""
    data = pd.Series([150, 200, 130, 190])
    result = iglu.hbgi(data)

    expected = 6.208971

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_hbgi_list_input():
    """Test HBGI calculation with list input."""
    data = [150, 200, 130, 190]
    result = iglu.hbgi(data)

    expected = 6.208971

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_hbgi_numpy_array_input():
    """Test HBGI calculation with numpy array input."""
    data = np.array([150, 200, 130, 190])
    result = iglu.hbgi(data)

    expected = 6.208971

    # Check output format
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

def test_hbgi_empty_data():
    """Test HBGI calculation with empty DataFrame."""
    data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.hbgi(data)


def test_hbgi_missing_values():
    """Test HBGI calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [150, np.nan, 200],
        }
    )

    result = iglu.hbgi(data)

    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, "HBGI"])
    assert len(result) == 1


def test_hbgi_all_below_threshold():
    """Test HBGI calculation when all values are below threshold."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [80, 90, 100],  # All values below 112.5
        }
    )

    result = iglu.hbgi(data)

    # Check that HBGI is 0 when all values are below threshold
    assert abs(result.loc[0, "HBGI"]) < 1e-10


def test_hbgi_all_above_threshold():
    """Test HBGI calculation when all values are above threshold."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [200, 250, 300],  # All values above 112.5
        }
    )

    result = iglu.hbgi(data)

    # Check that HBGI is positive when all values are above threshold
    assert result.loc[0, "HBGI"] > 0


def test_hbgi_multiple_subjects():
    """Test HBGI calculation with multiple subjects."""
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

    result = iglu.hbgi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result["id"]) == {"subject1", "subject2", "subject3"}

    # Check relative values
    # Subject 1 has lowest HBGI (all values below threshold)
    assert (
        result.loc[result["id"] == "subject1", "HBGI"].values[0]
        < result.loc[result["id"] == "subject2", "HBGI"].values[0]
    )
    # Subject 2 has highest HBGI (all values above threshold)
    assert (
        result.loc[result["id"] == "subject2", "HBGI"].values[0]
        > result.loc[result["id"] == "subject3", "HBGI"].values[0]
    )
    # Subject 3 has middle HBGI (mixed values)
    assert (
        result.loc[result["id"] == "subject3", "HBGI"].values[0]
        > result.loc[result["id"] == "subject1", "HBGI"].values[0]
    )


def test_hbgi_edge_cases():
    """Test HBGI calculation with edge case glucose values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:10:00",
                    "2020-01-01 00:15:00",
                ]
            ),
            "gl": [112.4, 112.5, 112.6, 500],  # Values around and above threshold
        }
    )

    result = iglu.hbgi(data)

    # Check that HBGI is calculated correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, "HBGI"])
    # HBGI should be positive but not extremely high
    assert 0 < result.loc[0, "HBGI"] < 100
