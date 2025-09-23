import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "median_glu"


def get_test_scenarios():
    """Get test scenarios for median glucose calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])

    # Filter scenarios for median_glu method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_median_glu_iglu_r_compatible(scenario):
    """Test median glucose calculation against expected results"""

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

    result_df = iglu.median_glu(df, **kwargs)

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


def test_median_glu_default():
    """Test median_glu with default parameters"""
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
                    "2020-01-01 00:00:00",  # 0 min
                    "2020-01-01 00:05:00",  # 5 min
                    "2020-01-01 00:10:00",  # 10 min
                    "2020-01-01 00:15:00",  # 15 min
                    "2020-01-01 00:00:00",  # subject2
                    "2020-01-01 00:05:00",  # subject2
                ]
            ),
            "gl": [150, 155, 160, 165, 140, 145],
        }
    )

    result = iglu.median_glu(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "median"])
    assert len(result) == 2  # One row per subject
    assert result["median"].iloc[0] == 157.5  # Median of [150, 155, 160, 165]
    assert result["median"].iloc[1] == 142.5  # Median of [140, 145]


def test_median_glu_series():
    """Test median_glu with Series input"""
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result = iglu.median_glu(series_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 152.5, rtol=1e-3)


def test_median_glu_empty():
    """Test median_glu with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.median_glu(empty_data)


def test_median_glu_constant_glucose():
    """Test median_glu with constant glucose values"""
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
            "gl": [150, 150, 150, 150],
        }
    )
    result = iglu.median_glu(data)
    assert len(result) == 1
    assert result["median"].iloc[0] == 150  # Should be 150 for constant glucose


def test_median_glu_missing_values():
    """Test median_glu with missing values"""
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
            "gl": [150, np.nan, 160, 165],
        }
    )
    result = iglu.median_glu(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["median"].iloc[0] == 160  # Median of [150, 160, 165]
