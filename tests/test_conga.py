import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "conga"


def get_test_scenarios():
    """Get test scenarios for CONGA calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for CONGA method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_conga_iglu_r_compatible(scenario):
    """Test CONGA calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.conga(df, **kwargs)

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


def test_conga_default():
    """Test CONGA with default parameters"""
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

    result = iglu.conga(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "CONGA"])
    assert all(pd.isna(result["CONGA"]))  # CONGA is not defined for data under 24 hours


def test_conga_series():
    """Test CONGA with Series input"""
    series_data = pd.Series(
        ([150, 155, 160, 165, 140, 145] * 2) * 10
    )  # 120 data points/10 hours
    result = iglu.conga(series_data, n=1)  # CONGA to be calculated for 1 hour
    assert isinstance(result, pd.DataFrame)
    assert "CONGA" in result.columns
    assert len(result) == 1


def test_conga_empty():
    """Test CONGA with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.conga(empty_data)


def test_conga_constant_glucose():
    """Test CONGA with constant glucose values"""
    series_data = pd.Series(
        ([150, 155, 160, 165, 140, 145] * 2) * 10
    )  # 120 data points/10 hours
    result = iglu.conga(series_data, n=1)  # CONGA to be calculated for 1 hour
    assert len(result) == 1
    assert result["CONGA"].iloc[0] == 0  # Should be 0 for constant glucose


def test_conga_missing_values():
    """Test CONGA with missing values"""
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
    result = iglu.conga(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_conga_different_n():
    """Test CONGA with different n values"""
    data_multi = pd.DataFrame(
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
            "gl": [150, 160, 170, 180, 190, 200, 210, 220],
        }
    )

    result_n1 = iglu.conga(data_multi, n=1)
    result_n2 = iglu.conga(data_multi, n=2)
    assert len(result_n1) == 1
    assert len(result_n2) == 1
    assert (
        result_n2["CONGA"].iloc[0] != result_n1["CONGA"].iloc[0]
    )  # Different n should give different results


def test_conga_timezone():
    """Test CONGA with timezone parameter"""
    data_multi = pd.DataFrame(
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
            "gl": [150, 160, 170, 180, 190, 200, 210, 220],
        }
    )

    result = iglu.conga(data_multi, tz="UTC")
    assert len(result) == 1
    assert isinstance(result["CONGA"].iloc[0], float)
