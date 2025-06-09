import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "auc"


def get_test_scenarios():
    """Get test scenarios for AUC calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for AUC method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_auc_iglu_r_compatible(scenario):
    """Test AUC calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if len(df) < 12:  # Need at least 12 records (1 hour at 5-min intervals) to calculate AUC
        pytest.skip("This AUC test requires at least few hours of data")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)


    result_df = iglu.auc(df, **kwargs)

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
        rtol=0.01,
    )


def test_auc_basic_output_format():
    """Test basic output format and structure of auc function"""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1", "subject1", "subject2", "subject2", "subject2"],
            "time": pd.to_datetime([
                "2020-01-01 00:00:00",  # 0 min
                "2020-01-01 00:05:00",  # 5 min
                "2020-01-01 00:10:00",  # 10 min
                "2020-01-01 00:15:00",  # 15 min
                "2020-01-01 00:00:00",  # subject2
                "2020-01-01 00:05:00",  # subject2
                "2020-01-01 00:10:00",  # subject2
            ]),
            "gl": [150, 155, 160, 165, 140, 145, 150],
        }
    )

    result = iglu.auc(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "hourly_auc"])
    assert all(result["hourly_auc"] >= 0)


def test_auc_empty_data():
    """Test auc function with empty DataFrame"""
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        iglu.auc(empty_data)


def test_auc_constant_glucose():
    """Test auc function with constant glucose values"""
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 14,        # need 13 points to workaround R bug in CGMS2DayByDay
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:15:00',
            '2020-01-01 00:20:00',
            '2020-01-01 00:25:00',
            '2020-01-01 00:30:00',
            '2020-01-01 00:35:00',
            '2020-01-01 00:40:00',
            '2020-01-01 00:45:00',
            '2020-01-01 00:50:00',
            '2020-01-01 00:55:00',
            '2020-01-01 01:00:00',
            '2020-01-01 01:05:00',
        ]),
        'gl': [100] * 14  # Constant glucose
    })
    result_single = iglu.auc(single_subject)
    assert len(result_single) == 1
    assert abs(result_single['hourly_auc'].iloc[0] - 100.0) < 0.001  # Should be equal to constant glucose * 60 min


def test_auc_missing_values():
    """Test auc function with missing glucose values"""
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:15:00'
        ]),
        'gl': [150, np.nan, 160, 165]
    })
    result_na = iglu.auc(data_with_na)
    assert isinstance(result_na, pd.DataFrame)
    assert len(result_na) == 1
    assert not np.isnan(result_na['hourly_auc'].iloc[0])  # Should handle NA values


def test_auc_multiple_days():
    """Test auc function with data spanning multiple days"""
    multi_day = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',  # Day 1
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-02 00:00:00', '2020-01-02 00:05:00',  # Day 2
            '2020-01-02 00:10:00', '2020-01-02 00:15:00'
        ]),
        'gl': [150, 155, 160, 165, 140, 145, 150, 155]
    })
    result_multi = iglu.auc(multi_day)
    assert isinstance(result_multi, pd.DataFrame)
    assert len(result_multi) == 1
    assert not np.isnan(result_multi['hourly_auc'].iloc[0])  # Should handle multiple days
