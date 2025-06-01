import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "range_glu"


def get_test_scenarios():
    """Get test scenarios for range_glu calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for range_glu method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.fixture
def test_data():
    """Fixture that provides test data for range_glu calculations"""
    return get_test_scenarios()


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_range_glu_iglu_r_compatible(scenario):
    """Test range_glu calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.range_glu(df, **kwargs)

    assert result_df is not None

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Compare DataFrames with precision to 0.001 for numeric columns
    pd.testing.assert_frame_equal(
        result_df.round(3),
        expected_df.round(3),
        check_dtype=False,  # Don't check dtypes since we might have different numeric types
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_names=True,
        check_exact=True,
        check_datetimelike_compat=True,
        check_categorical=True,
        check_like=True,
        check_freq=True,
        check_flags=True,
    )


def test_range_glu_basic_format():
    """Test basic output format and structure of range_glu function"""
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

    result = iglu.range_glu(data)

    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "range"])
    assert all(result["range"] >= 0)


def test_range_glu_series_input():
    """Test range_glu calculation with Series input"""
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result = iglu.range_glu(series_data)

    assert isinstance(result, pd.DataFrame)
    assert "range" in result.columns
    assert len(result) == 1
    assert result["range"].iloc[0] == 25  # max(165) - min(140)


def test_range_glu_empty_data():
    """Test range_glu calculation with empty DataFrame"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError, match="Data frame is empty"):
        iglu.range_glu(empty_data)


def test_range_glu_constant_glucose():
    """Test range_glu calculation with constant glucose values"""
    single_subject = pd.DataFrame(
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
            "gl": [150, 150, 150, 150],  # Constant glucose
        }
    )

    result = iglu.range_glu(single_subject)

    assert len(result) == 1
    assert result["range"].iloc[0] == 0  # Should be 0 for constant glucose


def test_range_glu_missing_values():
    """Test range_glu calculation with missing values"""
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
    result = iglu.range_glu(data_with_na)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["range"].iloc[0] == 15  # max(165) - min(150)


def test_range_glu_multiple_subjects():
    """Test range_glu calculation with multiple subjects"""
    multi_subject = pd.DataFrame(
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
    result = iglu.range_glu(multi_subject)

    assert len(result) == 2
    assert result.loc[result["id"] == "subject1", "range"].iloc[0] == 50  # 200 - 150
    assert result.loc[result["id"] == "subject2", "range"].iloc[0] == 60  # 190 - 130
