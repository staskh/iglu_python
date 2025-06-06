import json
import os

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "ea1c"


def get_test_scenarios():
    """Get test scenarios for eA1C calculations"""
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "expected_results.json"
    )
    if not os.path.exists(expected_results_path):
        pytest.skip("expected_results.json not found, skipping eA1C calculation test")
    try:
        with open(expected_results_path, "r") as f:
            expected_results = json.load(f)
    except Exception:
        pytest.skip(
            "expected_results.json could not be loaded, skipping eA1C calculation test"
        )
    # set local timezone if present
    if "config" in expected_results and "local_tz" in expected_results["config"]:
        try:
            iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
        except Exception:
            pass
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_ea1c_iglu_r_compatible(scenario):
    """Test eA1C calculation against expected results"""
    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.ea1c(df, **kwargs)

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
        check_exact=False,
        rtol=1e-3,
    )


def test_ea1c_default():
    """Test eA1C with default parameters"""
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

    result = iglu.ea1c(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "eA1C"])
    assert all(
        (result["eA1C"] >= 0) & (result["eA1C"] <= 20)
    )  # eA1C should be in reasonable range


def test_ea1c_series():
    """Test eA1C with Series input"""
    series_data = pd.Series([150, 155, 160, 165, 140, 145] * 10)  # 60 data points
    result = iglu.ea1c(series_data)
    assert isinstance(result, pd.DataFrame)
    assert "eA1C" in result.columns
    assert len(result) == 1
    assert (result["eA1C"].iloc[0] >= 0) & (result["eA1C"].iloc[0] <= 20)


def test_ea1c_empty():
    """Test eA1C with empty data"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.ea1c(empty_data)


def test_ea1c_constant_glucose():
    """Test eA1C with constant glucose values"""
    # Create data with constant glucose value
    constant_glucose = 150
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 24,  # 24 hours of data
            "time": pd.date_range(start="2020-01-01", periods=24, freq="H"),
            "gl": [constant_glucose] * 24,
        }
    )
    result = iglu.ea1c(data)
    assert len(result) == 1
    # eA1C should be close to the constant glucose value converted to A1c
    expected_ea1c = (constant_glucose + 46.7) / 28.7
    assert abs(result["eA1C"].iloc[0] - expected_ea1c) < 0.1


def test_ea1c_missing_values():
    """Test eA1C with missing values"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 24,
            "time": pd.date_range(start="2020-01-01", periods=24, freq="H"),
            "gl": [150, np.nan, 160, 165, 140, 145] * 4,  # Some missing values
        }
    )
    result = iglu.ea1c(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert not pd.isna(result["eA1C"].iloc[0])  # Should handle missing values


def test_ea1c_multiple_subjects():
    """Test eA1C with multiple subjects"""
    data = pd.DataFrame(
        {
            "id": [
                "subject1",
                "subject1",
                "subject2",
                "subject2",
                "subject3",
                "subject3",
            ],
            "time": pd.date_range(start="2020-01-01", periods=6, freq="H"),
            "gl": [150, 200, 130, 190, 140, 140],
        }
    )
    result = iglu.ea1c(data)
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "eA1C" in result.columns
    assert len(result) == 3  # Three subjects
    assert all((result["eA1C"] >= 0) & (result["eA1C"] <= 20))


def test_ea1c_extreme_values():
    """Test eA1C with extreme glucose values"""
    data = pd.DataFrame(
        {
            "id": ["subject1"] * 24,
            "time": pd.date_range(start="2020-01-01", periods=24, freq="H"),
            "gl": [40, 600] * 12,  # Alternating very low and very high values
        }
    )
    result = iglu.ea1c(data)
    assert len(result) == 1
    assert (result["eA1C"].iloc[0] >= 0) & (result["eA1C"].iloc[0] <= 20)
    # eA1C should be reasonable even with extreme values
    assert result["eA1C"].iloc[0] > 0
