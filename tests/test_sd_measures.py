import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "sd_measures"


def get_test_scenarios():
    """Get test scenarios for sd_measures calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for sd_measures method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_sd_measures_iglu_r_compatible(scenario):
    """Test sd_measures calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    result_df = iglu.sd_measures(df, **kwargs)

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
        rtol=0.001,
    )


def test_sd_measures_basic():
    """Test basic sd_measures functionality with simple data."""
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=48, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': np.random.normal(120, 20, 48)
    })
    
    result = iglu.sd_measures(data)
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]['id'] == 'subject1'
    
    # Check that all SD measures are present
    expected_columns = ['id', 'SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm']
    assert list(result.columns) == expected_columns
    
    # Check that all values are numeric and non-negative
    for col in ['SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm']:
        assert pd.notna(result.iloc[0][col])
        assert result.iloc[0][col] >= 0


def test_sd_measures_multiple_days():
    """Test with data spanning multiple days."""
    # Create 3 days of hourly data
    dates = pd.date_range('2020-01-01', periods=72, freq='1H')
    glucose_values = np.concatenate([
        np.random.normal(100, 15, 24),  # Day 1
        np.random.normal(130, 25, 24),  # Day 2  
        np.random.normal(110, 20, 24),  # Day 3
    ])
    
    data = pd.DataFrame({
        'id': ['subject1'] * 72,
        'time': dates,
        'gl': glucose_values
    })
    
    result = iglu.sd_measures(data)
    
    # All measures should be calculated
    assert not result.isnull().any().any()
    
    # SDdm should capture between-day variation
    assert result.iloc[0]['SDdm'] > 0


def test_sd_measures_constant_values():
    """Test with constant glucose values."""
    dates = pd.date_range('2020-01-01', periods=48, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': [120] * 48
    })
    
    result = iglu.sd_measures(data)
    
    # Most SD measures should be 0 or very small for constant values
    assert result.iloc[0]['SDw'] < 1e-10  # Should be essentially 0
    assert result.iloc[0]['SDhhmm'] < 1e-10
    assert result.iloc[0]['SDdm'] < 1e-10
    assert result.iloc[0]['SDb'] < 1e-10
    assert result.iloc[0]['SDbdm'] < 1e-10


def test_sd_measures_missing_values():
    """Test handling of missing glucose values."""
    dates = pd.date_range('2020-01-01', periods=48, freq='1H')
    glucose_values = np.random.normal(120, 20, 48)
    glucose_values[10:15] = np.nan  # Add some missing values
    
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': glucose_values
    })
    
    result = iglu.sd_measures(data)
    
    # Should still calculate values despite missing data
    assert not result.isnull().any().any()


def test_sd_measures_single_day():
    """Test with single day of data."""
    dates = pd.date_range('2020-01-01 00:00', periods=24, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 24,
        'time': dates,
        'gl': np.random.normal(120, 20, 24)
    })
    
    result = iglu.sd_measures(data)
    
    # SDdm and SDb should be NaN or 0 with only one day
    assert pd.isna(result.iloc[0]['SDdm']) or result.iloc[0]['SDdm'] == 0
    assert pd.isna(result.iloc[0]['SDb']) or result.iloc[0]['SDb'] == 0


def test_sd_measures_multiple_subjects_error():
    """Test that multiple subjects raise an error."""
    dates = pd.date_range('2020-01-01', periods=24, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 12 + ['subject2'] * 12,
        'time': dates,
        'gl': np.random.normal(120, 20, 24)
    })
    
    with pytest.raises(ValueError, match="Multiple subjects detected"):
        iglu.sd_measures(data)


def test_sd_measures_dt0_parameter():
    """Test with different dt0 (time frequency) values."""
    dates = pd.date_range('2020-01-01', periods=48, freq='30min')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': np.random.normal(120, 20, 48)
    })
    
    # Test with explicit dt0
    result_30min = iglu.sd_measures(data, dt0=30)
    result_auto = iglu.sd_measures(data)  # Should auto-detect 30min
    
    # Results should be similar (allowing for small numerical differences)
    for col in ['SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm']:
        assert abs(result_30min.iloc[0][col] - result_auto.iloc[0][col]) < 1e-10


def test_sd_measures_inter_gap_parameter():
    """Test the inter_gap parameter."""
    dates = pd.date_range('2020-01-01 00:00', periods=24, freq='1H')
    # Create a gap in the data
    dates_with_gap = pd.concat([
        pd.Series(dates[:10]),
        pd.Series(dates[15:])  # 5-hour gap
    ]).reset_index(drop=True)
    
    data = pd.DataFrame({
        'id': ['subject1'] * 19,
        'time': dates_with_gap,
        'gl': np.random.normal(120, 20, 19)
    })
    
    # Test with different inter_gap values
    result_small_gap = iglu.sd_measures(data, inter_gap=30)  # Small gap - won't interpolate
    result_large_gap = iglu.sd_measures(data, inter_gap=360)  # Large gap - will interpolate
    
    # Both should work but may give different results
    assert not result_small_gap.isnull().any().any()
    assert not result_large_gap.isnull().any().any()


def test_sd_measures_timezone_parameter():
    """Test the timezone parameter."""
    dates = pd.date_range('2020-01-01', periods=24, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 24,
        'time': dates,
        'gl': np.random.normal(120, 20, 24)
    })
    
    # Test with different timezone
    result_utc = iglu.sd_measures(data, tz="UTC")
    result_no_tz = iglu.sd_measures(data)
    
    # Results should be the same regardless of timezone for this data
    for col in ['SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm']:
        assert abs(result_utc.iloc[0][col] - result_no_tz.iloc[0][col]) < 1e-10


def test_sd_measures_empty_dataframe():
    """Test that empty DataFrame raises appropriate error."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    
    with pytest.raises(ValueError):
        iglu.sd_measures(data)


def test_sd_measures_output_dtypes():
    """Test that output has correct data types."""
    dates = pd.date_range('2020-01-01', periods=48, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': np.random.normal(120, 20, 48)
    })
    
    result = iglu.sd_measures(data)
    
    # Check data types
    assert result['id'].dtype == object  # string
    for col in ['SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm']:
        assert np.issubdtype(result[col].dtype, np.number)


def test_sd_measures_reproducibility():
    """Test that results are reproducible with same input."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=48, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': np.random.normal(120, 20, 48)
    })
    
    result1 = iglu.sd_measures(data)
    result2 = iglu.sd_measures(data)
    
    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2) 