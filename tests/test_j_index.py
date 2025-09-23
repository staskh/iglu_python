import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "j_index"


def get_test_scenarios():
    """Get test scenarios for J-index calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])

    # Filter scenarios for J-index method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_j_index_iglu_r_compatible(scenario):
    """Test J-index calculation against expected results"""

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

    result_df = iglu.j_index(df, **kwargs)

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


def test_j_index_basic():
    """Test basic functionality of j_index"""

    # Create test data with known values
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
            "gl": [150, 200, 130, 190],
        }
    )

    # Test with DataFrame input
    result = iglu.j_index(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "J_index" in result.columns
    assert len(result) == 2  # Two subjects

    # Check calculations
    # For subject1: mean = 175, sd = 25, J-index = 0.001 * (175 + 25)*2 ~~ 40.000
    # For subject2: mean = 160, sd = 30, J-index = 0.001 * (160 + 30)*2 ~~ 38.000
    np.testing.assert_allclose(result.loc[result["id"] == "subject1", "J_index"].iloc[0], 44.249369, rtol=1e-3)
    np.testing.assert_allclose(result.loc[result["id"] == "subject2", "J_index"].iloc[0], 40.97645, rtol=1e-3)


    # Test with Series input
    result_series = iglu.j_index(data["gl"])

    # Check output format for Series input
    assert isinstance(result_series, float)
    np.testing.assert_allclose(result_series, 40.216444, rtol=1e-3)


def test_j_index_empty_data():
    """Test j_index with empty data"""

    # Empty DataFrame
    data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.j_index(data)


def test_j_index_missing_values():
    """Test j_index with missing values"""

    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [150, np.nan, 200],
        }
    )

    result = iglu.j_index(data)

    # Check that missing values are handled appropriately
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert not np.isnan(result["J_index"].iloc[0])

    # For subject1: mean = 175, sd = 25, J-index = 0.001 * (175 + 25)**2 ~~ 45.000
    assert (abs(result["J_index"].iloc[0] - 45.000) / 45.000) < 0.1


def test_j_index_constant_values():
    """Test j_index with constant glucose values"""

    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [150, 150, 150],
        }
    )

    result = iglu.j_index(data)

    # For constant values: mean = 150, sd = 0, J-index = 0.001 * (150 + 0)**2 ~~ 22.5
    assert (abs(result["J_index"].iloc[0] - 22.5) / 22.5) < 0.1
