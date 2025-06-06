import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "quantile_glu"


def get_test_scenarios():
    """Get test scenarios for quantile calculation"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for quantile_glu method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_quantile_glu_iglu_r_compatible(scenario):
    """Test quantile calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.quantile_glu(df, **kwargs)

    assert result_df is not None

    # Compare DataFrames with precision to 0.001 for numeric columns
    pd.testing.assert_frame_equal(
        result_df.round(3),
        expected_df.round(3),
        check_dtype=False,  # Don't check dtypes since we might have different numeric types
        check_index_type=True,
        check_column_type=False,  # skip checking column types
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


def test_quantile_glu_default():
    """Test quantile calculation with default parameters"""
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

    result = iglu.quantile_glu(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", '0', '25', '50', '75', '100'])
    assert len(result) == 2  # One row per subject


def test_quantile_glu_series():
    """Test quantile calculation with Series input"""
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result = iglu.quantile_glu(series_data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in [0.0, 25.0, 50.0, 75.0, 100.0])
    assert len(result) == 1


def test_quantile_glu_empty():
    """Test quantile calculation with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.quantile_glu(empty_data)


def test_quantile_glu_custom_quantiles():
    """Test quantile calculation with custom quantiles"""
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
            "gl": [150, 155, 160, 165],
        }
    )
    custom_quantiles = [0, 33, 66, 100]
    result = iglu.quantile_glu(data, quantiles=custom_quantiles)
    assert all(col in result.columns for col in ["id"] + [str(q) for q in custom_quantiles])
    assert len(result) == 1


def test_quantile_glu_missing_values():
    """Test quantile calculation with missing values"""
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
    result = iglu.quantile_glu(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    # Check that NA values are properly handled
    assert not result.isna().any().any()


def test_quantile_glu_single_value():
    """Test quantile calculation with single value per subject"""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject2"],
            "time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:00"]),
            "gl": [150, 160],
        }
    )
    result = iglu.quantile_glu(data)
    assert len(result) == 2
    # For single values, all quantiles should be the same
    assert all(result.iloc[0, 1:] == 150)  # All quantiles for subject1 should be 150
    assert all(result.iloc[1, 1:] == 160)  # All quantiles for subject2 should be 160


def test_quantile_glu_constant_values():
    """Test quantile calculation with constant values"""
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
    result = iglu.quantile_glu(data)
    assert len(result) == 1
    # For constant values, all quantiles should be the same
    assert all(result.iloc[0, 1:] == 150)
