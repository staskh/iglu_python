import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "roc"


def get_test_scenarios():
    """Get test scenarios for ROC calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for ROC method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_roc_iglu_r_compatible(scenario):
    """Test ROC calculation against expected results"""

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

    result_df = iglu.roc(df, **kwargs)

    assert result_df is not None

    # accommodating R implementation format
    result_df.drop(columns=["time"], inplace=True)
    # drop all nan values from both dataframes to remove dependencies on CGMS2DayByDay "last day" error
    result_df = result_df.dropna().reset_index(drop=True)
    expected_df = expected_df.dropna().reset_index(drop=True)
    # if result_df.shape[0] < expected_df.shape[0]:
    #     expected_df = expected_df.iloc[: result_df.shape[0]]

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


def test_roc_default():
    """Test ROC with default parameters"""
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
            "gl": [150, 160, 170, 180, 140, 145],
        }
    )

    result = iglu.roc(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "time", "roc"])
    assert len(result) > 0  # Should have ROC values for each time point
    assert result["roc"].isna().any()  # Should have some NaN values at the start


def test_roc_series():
    """Test ROC with Series input"""
    series_data = pd.Series([150, 160, 170, 180, 190, 200],
            index=pd.date_range(start="2020-01-01 00:00:00", periods=6, freq="5min"))
    result = iglu.roc(series_data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "time", "roc"])
    assert len(result) > 0
    assert result["roc"].isna().any()


def test_roc_empty():
    """Test ROC with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.roc(empty_data)


def test_roc_constant_glucose():
    """Test ROC with constant glucose values"""
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

    result = iglu.roc(data)
    assert len(result) > 0
    # After the initial NaN values, ROC should be 0 for constant glucose
    assert np.allclose(result["roc"].dropna(), 0, atol=1e-10)


def test_roc_missing_values():
    """Test ROC with missing values"""
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
    result = iglu.roc(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert result["roc"].isna().any()


def test_roc_different_timelag():
    """Test ROC with different timelag values"""
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
            "gl": [150, 160, 170, 180, 190, 200, 210, 220],
        }
    )

    result_15 = iglu.roc(data, timelag=15)
    result_30 = iglu.roc(data, timelag=30)

    assert len(result_15) > 0
    assert len(result_30) > 0
    # Different timelag should give different ROC values
    assert not np.array_equal(result_15["roc"].dropna(), result_30["roc"].dropna())


def test_roc_timezone():
    """Test ROC with timezone parameter"""
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
            "gl": [150, 160, 170, 180, 190, 200, 210, 220],
        }
    )

    result = iglu.roc(data, tz="UTC")
    assert len(result) > 0
    assert isinstance(result["time"].iloc[0], pd.Timestamp)
    assert result["time"].iloc[0].tzinfo is not None
