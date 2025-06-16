import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "modd"


def get_test_scenarios():
    """Get test scenarios for modd calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # Filter scenarios for modd method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.fixture
def test_data():
    """Fixture that provides test data for modd calculations"""
    return get_test_scenarios()


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_modd_iglu_r_compatible(scenario):
    """Test modd calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.modd(df, **kwargs)

    assert result_df is not None

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})


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


def test_modd_default_output():
    """Test modd calculation with default parameters"""
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
            "gl": [150, 200, 180, 160, 140, 190],
        }
    )

    result = iglu.modd(data)

    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "MODD"])
    assert all(pd.isna(result["MODD"]))  # Should be NaN for insufficient data


def test_modd_custom_lag():
    """Test modd calculation with custom lag value"""
    samples_per_day = int(24*60/5)  # sample each 5 min
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 3 * samples_per_day,
            "time": pd.date_range(start="2020-01-01 00:00:00", periods=3*samples_per_day, freq="5min"),
            "gl": [150]*samples_per_day + [200]*samples_per_day + [180]*samples_per_day,
        }
    )

    result = iglu.modd(data, lag=2)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "MODD"])
    assert all(result["MODD"] >= 0)


def test_modd_series_input():
    """Test modd calculation with Series input"""
    samples_per_day = int(24*60/5)  # sample each 5 min
    series_data = pd.Series(
        [150]*samples_per_day + [200]*samples_per_day + [250]*samples_per_day,
        index=pd.date_range(start="2020-01-01 00:00:00", periods=3*samples_per_day, freq="5min")
    )
    result = iglu.modd(series_data)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 50.0, rtol=1e-3)

    # Exception for series without DatetimeIndex
    with pytest.raises(ValueError):
        iglu.modd(series_data.reset_index(drop=True))


def test_modd_empty_input():
    """Test modd calculation with empty DataFrame"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.modd(empty_data)
