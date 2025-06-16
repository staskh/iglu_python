import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "pgs"


def get_test_scenarios():
    """Get test scenarios for PGS calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for pgs method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_pgs_iglu_r_compatible(scenario):
    """Test PGS calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.pgs(df, **kwargs)

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


def test_pgs_default():
    """Test PGS with default parameters"""
    data = pd.DataFrame(
        {
            "id": [
                "subject1",
                "subject1",
                "subject1",
                "subject1",
                "subject2",
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
                    "2020-01-01 00:10:00",  # subject2
                ]
            ),
            "gl": [150, 155, 160, 165, 140, 145, 150],
        }
    )

    result = iglu.pgs(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "PGS"])
    assert len(result) == 2  # One row per subject
    assert all(result["PGS"] > 0)  # PGS should always be positive


def test_pgs_series():
    """Test PGS with Series input"""
    series_data = pd.Series(
        [150, 155, 160, 165, 140, 145],
        index=pd.date_range(
            start="2020-01-01 00:00:00",
            periods=6,
            freq="5min"
        )
    )
    result = iglu.pgs(series_data)
    expected = 16.494037
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-3)

    # Exception for series without DatetimeIndex
    with pytest.raises(ValueError):
        iglu.pgs(series_data.reset_index(drop=True))



def test_pgs_empty():
    """Test PGS with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.pgs(empty_data)


def test_pgs_constant_glucose():
    """Test PGS with constant glucose values"""
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
    result = iglu.pgs(data)
    assert len(result) == 1
    assert result["PGS"].iloc[0] > 0  # PGS should always be positive


def test_pgs_missing_values():
    """Test PGS with missing values"""
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
    result = iglu.pgs(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["PGS"].iloc[0] > 0  # PGS should always be positive


def test_pgs_different_durations():
    """Test PGS with different duration parameters"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 11,
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
                    "2020-01-01 00:40:00",
                    "2020-01-01 00:45:00",
                    "2020-01-01 00:50:00",
                ]
            ),
            "gl": [150, 160, 60, 60, 60, 180, 190, 200, 210, 220, 230],
        }
    )

    result_default = iglu.pgs(data, dur_length=10, end_length=10)
    result_custom = iglu.pgs(data, dur_length=40, end_length=40)
    assert len(result_default) == 1
    assert len(result_custom) == 1
    assert (
        result_default["PGS"].iloc[0] != result_custom["PGS"].iloc[0]
    )  # Different durations should give different results


def test_pgs_extreme_values():
    """Test PGS with extreme glucose values"""
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
            "gl": [40, 400, 40, 400],  # Extreme hypo and hyper values
        }
    )
    result = iglu.pgs(data)
    assert len(result) == 1
    assert result["PGS"].iloc[0] > 0  # PGS should always be positive
    assert result["PGS"].iloc[0] > 10  # Extreme values should give high PGS scores
