import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "mage"


def get_test_scenarios():
    """Get test scenarios for MAGE calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)

    # Filter scenarios for MAGE method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.fixture
def test_data():
    """Fixture that provides test data for MAGE calculations"""
    return get_test_scenarios()


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_mage_iglu_r_compatible(scenario):
    """Test MAGE calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})

    expected_df = expected_df.dropna(subset=["MAGE"])
    if expected_df.empty:
        pytest.skip("This MAGE test has no numeric value to compare")

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    result_df = iglu.mage(df, **kwargs)

    assert result_df is not None

    result_df = result_df[["id", "MAGE"]]

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
        rtol=0.001,
    )


@pytest.fixture
def base_data():
    """Fixture providing base test data for MAGE calculations"""
    return pd.DataFrame(
        {
            "id": ["subject1"] * 8 + ["subject2"]*2,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:20:00",
                    "2020-01-01 00:40:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 01:20:00",
                    "2020-01-01 01:40:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 02:20:00",

                    "2020-01-01 00:00:00",  # 0 min
                    "2020-01-01 00:05:00",  # 5 min

                ]
            ),
            "gl": [150, 200, 180, 160, 140, 190, 170, 210] + [150, 200],
        }
    )


@pytest.fixture
def multi_point_data():
    """Fixture providing multi-point test data for MAGE calculations"""
    return pd.DataFrame(
        {
            "id": ["subject1"] * 8,
            "time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:20:00",
                    "2020-01-01 00:40:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 01:20:00",
                    "2020-01-01 01:40:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 02:20:00",
                ]
            ),
            "gl": [150, 200, 180, 160, 140, 190, 170, 210],
        }
    )


def test_mage_default_parameters(base_data):
    """Test MAGE calculation with default parameters"""
    result = iglu.mage(base_data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "MAGE"])
    assert result.iloc[0]["MAGE"] >= 0
    assert np.isnan(result.iloc[1]["MAGE"])

def test_mage_naive_version(base_data):
    """Test MAGE calculation with naive version"""
    result = iglu.mage(base_data, version="naive")
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["id", "MAGE"])
    assert result.iloc[0]["MAGE"] >= 0
    assert np.isnan(result.iloc[1]["MAGE"])


def test_mage_series_without_datetime_index():
    """Test MAGE calculation with Series input"""
    series_data = pd.Series([150, 200, 180, 160, 140, 190])
    with pytest.raises(ValueError):
        iglu.mage(series_data)


def test_mage_series_with_datetime_index():
    """Test MAGE calculation with Series input"""
    series_data = pd.Series([150, 200, 180, 160, 140, 190], index=pd.date_range(start="2020-01-01 00:00:00", periods=6, freq="5min"))
    result = iglu.mage(series_data)
    assert isinstance(result, (float,np.float64))

def test_mage_empty_data():
    """Test MAGE calculation with empty DataFrame"""
    empty_data = pd.DataFrame(columns=["id", "time", "gl"])
    with pytest.raises(ValueError):
        iglu.mage(empty_data)


def test_mage_constant_glucose():
    """Test MAGE calculation with constant glucose values"""
    n_measurements = 12*24
    single_subject = pd.DataFrame(
        {
            "id": ["subject1"] * n_measurements,
            "time": pd.date_range(start="2020-01-01 00:00:00", periods=n_measurements, freq="5min"),
            "gl": [150]*n_measurements,
        }
    )
    result = iglu.mage(single_subject)
    assert len(result) == 1
    assert np.isnan(result["MAGE"].iloc[0])


def test_mage_missing_values():
    """Test MAGE calculation with missing values"""
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
            "gl": [150, np.nan, 180, 160],
        }
    )
    result = iglu.mage(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_mage_direction_plus(multi_point_data):
    """Test MAGE calculation with direction='plus'"""
    result = iglu.mage(multi_point_data, direction="plus")
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_direction_minus(multi_point_data):
    """Test MAGE calculation with direction='minus'"""
    result = iglu.mage(multi_point_data, direction="minus")
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_direction_max(multi_point_data):
    """Test MAGE calculation with direction='max'"""
    result = iglu.mage(multi_point_data, direction="max")
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_moving_average_windows(multi_point_data):
    """Test MAGE calculation with custom moving average windows"""
    result = iglu.mage(multi_point_data, short_ma=3, long_ma=6)
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_return_type_df(multi_point_data):
    """Test MAGE calculation with return_type='df'"""
    result = iglu.mage(multi_point_data, return_type="df")
    assert isinstance(result, pd.DataFrame)
    assert all(
        col in result.columns
        for col in ["id", "start", "end", "MAGE", "plus_or_minus", "first_excursion"]
    )


def test_mage_timezone(multi_point_data):
    """Test MAGE calculation with timezone parameter"""
    result = iglu.mage(multi_point_data, tz="UTC")
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_inter_gap(multi_point_data):
    """Test MAGE calculation with custom inter_gap"""
    result = iglu.mage(multi_point_data, inter_gap=60)
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_max_gap(multi_point_data):
    """Test MAGE calculation with custom max_gap"""
    result = iglu.mage(multi_point_data, max_gap=120)
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_sd_multiplier(multi_point_data):
    """Test MAGE calculation with custom sd_multiplier for naive version"""
    result = iglu.mage(multi_point_data, version="naive", sd_multiplier=1.5)
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_swapped_ma_windows(multi_point_data):
    """Test MAGE calculation with swapped short_ma and long_ma values"""
    result = iglu.mage(multi_point_data, short_ma=32, long_ma=5)
    assert len(result) == 1
    assert isinstance(result["MAGE"].iloc[0], float)


def test_mage_insufficient_data():
    """Test MAGE calculation with insufficient data points"""
    small_data = pd.DataFrame(
        {
            "id": ["subject1"] * 2,
            "time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00"]
            ),
            "gl": [150, 160],
        }
    )
    result = iglu.mage(small_data)
    assert len(result) == 1
    assert pd.isna(result["MAGE"].iloc[0])
