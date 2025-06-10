import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "active_percent"


def get_test_scenarios():
    """Get test scenarios for active_percent calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for active_percent method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]

@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_active_percent_iglu_r_compatible(scenario):
    """Test active_percent calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])


    result_df = iglu.active_percent(df, **kwargs)

    assert result_df is not None

    # Convert timestamp columns to strings for comparison
    for col in ["start_date", "end_date"]:
        if col in result_df.columns:
            # drop tz information
            result_df[col] = result_df[col].apply(lambda x: x.isoformat())

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
        rtol=0.01,
    )


def test_active_percent_basic_output():
    """Test basic output format and structure of active_percent function"""
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
                    "2020-01-01 00:05:00",  # 5 min (gap)
                    "2020-01-01 00:15:00",  # 15 min
                    "2020-01-01 00:20:00",  # 20 min
                    "2020-01-01 00:00:00",  # subject2
                    "2020-01-01 00:05:00",  # subject2
                ]
            ),
            "gl": [150, np.nan, 160, 165, 140, 145],
        }
    )

    result = iglu.active_percent(data)

    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(
        col in result.columns
        for col in ["id", "active_percent", "ndays", "start_date", "end_date"]
    )
    assert all((result["active_percent"] >= 0) & (result["active_percent"] <= 100))
    assert all(result["ndays"] >= 0)

def test_active_percent_custom_dt0():
    """Test active_percent with custom dt0 parameter"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 4,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:15:00",
                    "2020-01-01 00:20:00",
                ]
            ),
            "gl": [150, np.nan, 160, 165],
        }
    )

    result = iglu.active_percent(data, dt0=5)
    assert isinstance(result, pd.DataFrame)

def test_active_percent_consistent_end_date():
    """Test active_percent with consistent end date"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 4,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:15:00",
                    "2020-01-01 00:20:00",
                ]
            ),
            "gl": [150, np.nan, 160, 165],
        }
    )

    end_date = datetime(2020, 1, 1, 1, 0)  # 1 hour after start
    result = iglu.active_percent(data, range_type='manual', consistent_end_date=end_date)
    assert all(result["end_date"].dt.tz_localize(None) == pd.to_datetime(end_date).tz_localize(None))

def test_active_percent_timezone():
    """Test active_percent with timezone parameter"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 4,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                    "2020-01-01 00:15:00",
                    "2020-01-01 00:20:00",
                ]
            ),
            "gl": [150, np.nan, 160, 165],
        }
    )

    result = iglu.active_percent(data, tz="GMT")
    assert isinstance(result, pd.DataFrame)

def test_active_percent_empty_data():
    """Test active_percent with empty DataFrame"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.active_percent(empty_data)

def test_active_percent_single_subject_no_gaps():
    """Test active_percent with single subject and no gaps"""
    single_subject = pd.DataFrame(
        {
            "id": ["subject1"] * 3,
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "gl": [150, 155, 160],
        }
    )
    
    result = iglu.active_percent(single_subject, dt0=5)
    assert len(result) == 1
    assert result["active_percent"].iloc[0] == 100.0  # Should be 100% active with no gaps