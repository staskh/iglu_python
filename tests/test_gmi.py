import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "gmi"


def get_test_scenarios():
    """Get test scenarios for GMI calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    
    # Filter scenarios for GMI method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_gmi_iglu_r_compatible(scenario):
    """Test GMI calculation against expected results from R implementation"""

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

    result_df = iglu.gmi(df, **kwargs)

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
        rtol=0.001,
    )


def test_gmi_basic():
    """Test basic GMI calculation with known glucose values"""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1", "subject1", "subject2", "subject2"],
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
            "gl": [150, 160, 170, 180, 140, 145],
        }
    )

    result = iglu.gmi(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "GMI"])
    assert len(result) == 2  # One row per subject

    # Calculate expected GMI for subject1
    # Mean glucose = (150 + 160 + 170 + 180) / 4 = 165
    # GMI = 3.31 + (0.02392 * 165) = 7.2568
    expected_gmi1 = 3.31 + (0.02392 * 165)
    assert abs(result.loc[result["id"] == "subject1", "GMI"].iloc[0] - expected_gmi1) < 0.001

    # Calculate expected GMI for subject2
    # Mean glucose = (140 + 145) / 2 = 142.5
    # GMI = 3.31 + (0.02392 * 142.5) = 6.7186
    expected_gmi2 = 3.31 + (0.02392 * 142.5)
    assert abs(result.loc[result["id"] == "subject2", "GMI"].iloc[0] - expected_gmi2) < 0.001


def test_gmi_series():
    """Test GMI with Series input"""
    series_data = pd.Series([150, 160, 170, 180, 190, 200])
    result = iglu.gmi(series_data)
    assert isinstance(result, (float,np.float64))

    # Calculate expected GMI
    # Mean glucose = (150 + 160 + 170 + 180 + 190 + 200) / 6 = 175
    # GMI = 3.31 + (0.02392 * 175) = 7.496
    expected_gmi = 3.31 + (0.02392 * 175)
    np.testing.assert_allclose(result, expected_gmi, rtol=0.001)


def test_gmi_empty():
    """Test GMI with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.gmi(empty_data)


def test_gmi_constant_glucose():
    """Test GMI with constant glucose values"""
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
            "gl": [150] * 6,  # Constant glucose
        }
    )

    result = iglu.gmi(data)
    assert len(result) == 1
    # For constant glucose of 150, GMI should be 3.31 + (0.02392 * 150) = 6.898
    expected_gmi = 3.31 + (0.02392 * 150)
    assert abs(result["GMI"].iloc[0] - expected_gmi) < 0.001


def test_gmi_missing_values():
    """Test GMI with missing values"""
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
            "gl": [150, np.nan, 170, 180],
        }
    )
    result = iglu.gmi(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    # Mean glucose = (150 + 170 + 180) / 3 = 166.67
    # GMI = 3.31 + (0.02392 * 166.67) = 7.2987
    expected_gmi = 3.31 + (0.02392 * 166.67)
    assert abs(result["GMI"].iloc[0] - expected_gmi) < 0.001


def test_gmi_extreme_values():
    """Test GMI with extreme glucose values"""
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
            "gl": [40, 400, 600, 800],  # Extreme values
        }
    )
    result = iglu.gmi(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    # Mean glucose = (40 + 400 + 600 + 800) / 4 = 460
    # GMI = 3.31 + (0.02392 * 460) = 14.3132
    expected_gmi = 3.31 + (0.02392 * 460)
    assert abs(result["GMI"].iloc[0] - expected_gmi) < 0.001
