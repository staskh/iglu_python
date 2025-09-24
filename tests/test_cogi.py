import json
import os

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu


def test_cogi_basic():
    """Test basic cogi calculation with known values."""
    # Create test data with known glucose values
    data = pd.DataFrame(
        {
            "id": [
                "subject1",
                "subject1",
                "subject1",
                "subject2",
                "subject2",
                "subject2",
            ],
            "time": pd.date_range(start="2020-01-01", periods=6, freq="5min"),
            "gl": [150, 200, 180, 130, 180, 160],
        }
    )

    # Calculate cogi
    result = iglu.cogi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "COGI" in result.columns
    assert len(result) == 2  # Two subjects

    # Check that COGI values are between 0 and 100 (percentage)
    assert all((result["COGI"] >= 0) & (result["COGI"] <= 100))

    # Check that subject2 has higher COGI than subject1
    # (since subject2 has more values in range and less variability)
    subject1_cogi = result[result["id"] == "subject1"]["COGI"].iloc[0]
    subject2_cogi = result[result["id"] == "subject2"]["COGI"].iloc[0]
    assert subject2_cogi > subject1_cogi


def test_cogi_series_input():
    """Test cogi calculation with Series input."""
    # Create test data as Series
    data = pd.Series([150, 200, 180, 130, 190, 160])

    # Calculate cogi
    result = iglu.cogi(data)

    # Check output format
    assert isinstance(result, (float,np.float64))

    # Check that COGI value is between 0 and 100
    np.testing.assert_allclose(result, 81.934259, rtol=0.001)


def test_cogi_custom_parameters():
    """Test cogi calculation with custom parameters."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, 200, 180],
        }
    )

    # Test with custom parameters
    result = iglu.cogi(data, targets=[80, 150], weights=[0.3, 0.6, 0.1])

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "COGI" in result.columns
    assert len(result) == 1

    # Check that COGI value is between 0 and 100
    assert (result["COGI"].iloc[0] >= 0) and (result["COGI"].iloc[0] <= 100)


def test_cogi_empty_data():
    """Test cogi calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=["id", "time", "gl"])

    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        iglu.cogi(data)


def test_cogi_missing_values():
    """Test cogi calculation with missing values."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [150, np.nan, 180],
        }
    )

    # Calculate cogi
    result = iglu.cogi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "COGI" in result.columns
    assert len(result) == 1

    # Check that COGI value is between 0 and 100
    assert (result["COGI"].iloc[0] >= 0) and (result["COGI"].iloc[0] <= 100)


def test_cogi_all_in_range():
    """Test cogi calculation with all values in range."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [100, 110, 120],  # All values between 70 and 180
        }
    )

    # Calculate cogi
    result = iglu.cogi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "COGI" in result.columns
    assert len(result) == 1

    # Check that COGI value is high (good control)
    assert result["COGI"].iloc[0] > 80


def test_cogi_all_below_range():
    """Test cogi calculation with all values below range."""
    data = pd.DataFrame(
        {
            "id": ["subject1", "subject1", "subject1"],
            "time": pd.date_range(start="2020-01-01", periods=3, freq="5min"),
            "gl": [50, 60, 65],  # All values below 70
        }
    )

    # Calculate cogi
    result = iglu.cogi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "COGI" in result.columns
    assert len(result) == 1

    # Check that COGI value is low (poor control)
    assert result["COGI"].iloc[0] < 50


def test_cogi_multiple_subjects():
    """Test cogi calculation with multiple subjects."""
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
            "time": pd.date_range(start="2020-01-01", periods=6, freq="5min"),
            "gl": [150, 200, 130, 190, 140, 140],
        }
    )

    # Calculate cogi
    result = iglu.cogi(data)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "COGI" in result.columns
    assert len(result) == 3  # Three subjects

    # Check that COGI values are between 0 and 100
    assert all((result["COGI"] >= 0) & (result["COGI"] <= 100))

    # Check that subject3 has higher COGI than others (since values are more stable)
    subject3_cogi = result[result["id"] == "subject3"]["COGI"].iloc[0]
    subject1_cogi = result[result["id"] == "subject1"]["COGI"].iloc[0]
    subject2_cogi = result[result["id"] == "subject2"]["COGI"].iloc[0]
    assert subject3_cogi >= subject1_cogi
    assert subject3_cogi >= subject2_cogi


def get_cogi_test_scenarios():
    """Get test scenarios for COGI calculations"""
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "expected_results.json"
    )
    if not os.path.exists(expected_results_path):
        pytest.skip("expected_results.json not found, skipping COGI calculation test")
    try:
        with open(expected_results_path, "r") as f:
            expected_results = json.load(f)
    except Exception:
        pytest.skip(
            "expected_results.json could not be loaded, skipping COGI calculation test"
        )

    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
 
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == "cogi"
    ]


@pytest.mark.parametrize("scenario", get_cogi_test_scenarios())
def test_cogi_iglu_r_compatible(scenario):
    """Test COGI calculation against expected results"""
    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    result_df = iglu.cogi(df, **kwargs)

    assert result_df is not None

    # Compare DataFrames with precision to 0.01 for numeric columns
    pd.testing.assert_frame_equal(
        result_df,
        expected_df,
        check_dtype=False,
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
