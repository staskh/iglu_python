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
    # pd.set_option('future.no_silent_downcasting', True)
    expected_df = expected_df.replace({None: np.nan})

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
    hours = 48
    dt0 = 5
    samples = int(hours * 60/dt0)
    dates = pd.date_range('2020-01-01', periods=samples, freq='5min')
    data = pd.DataFrame({
        'id': ['subject1'] * samples,
        'time': dates,
        'gl': np.random.normal(120, 20, samples)
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
    days = 3
    dt0 = 5
    samples = int(days*24*60/dt0)
    dates = pd.date_range('2020-01-01', periods=samples, freq=f"{dt0}min")
    glucose_values = np.concatenate([
        np.random.normal(100, 15, int(samples/3)),  # Day 1
        np.random.normal(130, 25, int(samples/3)),  # Day 2
        np.random.normal(110, 20, int(samples/3)),  # Day 3
    ])

    data = pd.DataFrame({
        'id': ['subject1'] * samples,
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
    dates = pd.date_range('2020-01-01', periods=48, freq='1h')
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


def test_sd_measures_single_day():
    """Test with single day of data."""
    dates = pd.date_range('2020-01-01 00:00', periods=24, freq='1h')
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
    hours = 24
    dt0 = 5
    samples = int(hours*60/dt0)
    half_samples = int(samples/2)
    dates = pd.date_range('2020-01-01', periods=samples, freq=f"{dt0}min")
    data = pd.DataFrame({
        'id': ['subject1'] * half_samples + ['subject2'] * half_samples,
        'time': dates,
        'gl': np.random.normal(120, 20, samples)
    })

    result = iglu.sd_measures(data)
    # Test that multiple subjects are handled correctly
    assert len(result) == 2  # Should have results for both subjects
    assert set(result['id']) == {'subject1', 'subject2'}  # Should have both subject IDs

    # Test that results are calculated for each subject independently
    subject1_data = data[data['id'] == 'subject1']
    subject2_data = data[data['id'] == 'subject2']

    result1 = iglu.sd_measures(subject1_data)
    result2 = iglu.sd_measures(subject2_data)

    # Results from individual calculations should match combined results
    for col in ['SDw', 'SDhhmm', 'SDwsh']:
        assert abs(result.iloc[0][col] - result1.iloc[0][col]) < 1e-10  # subject1
        assert abs(result.iloc[1][col] - result2.iloc[0][col]) < 1e-10  # subject2

    for col in ['SDdm', 'SDb', 'SDbdm']:
        assert np.isnan(result.iloc[0][col])
        assert np.isnan(result.iloc[1][col])


def test_sd_measures_dt0_parameter():
    """Test with different dt0 (time frequency) values."""
    hours = 48
    dt0 = 30
    samples = int(hours * 60/dt0)
    dates = pd.date_range('2020-01-01', periods=samples, freq=f'{dt0}min')
    data = pd.DataFrame({
        'id': ['subject1'] * samples,
        'time': dates,
        'gl': np.random.normal(120, 20, samples)
    })

    # Test with explicit dt0
    result_30min = iglu.sd_measures(data, dt0=30)
    result_auto = iglu.sd_measures(data)  # Should auto-detect 30min

    # Results should be similar (allowing for small numerical differences)
    for col in ['SDw', 'SDhhmm', 'SDwsh', 'SDdm', 'SDb', 'SDbdm']:
        assert abs(result_30min.iloc[0][col] - result_auto.iloc[0][col]) < 1e-10


def test_sd_measures_inter_gap_parameter():
    """Test the inter_gap parameter."""
    hours = 48
    dt0 = 30
    samples = int(hours * 60/dt0)
    dates = pd.date_range('2020-01-01', periods=samples, freq=f'{dt0}min')
    data = pd.DataFrame({
        'id': ['subject1'] * samples,
        'time': dates,
        'gl': np.random.normal(120, 20, samples)
    })

    #make 5h gap from 12:00
    gap_start = 12*(60/dt0)
    gap_hour = int(60/dt0)
    data.loc[gap_start:gap_start + 5*gap_hour - 1, 'gl'] = np.nan

    # Test with different inter_gap values
    result_small_gap = iglu.sd_measures(data, inter_gap=gap_hour)  # Small gap - won't interpolate
    result_large_gap = iglu.sd_measures(data, inter_gap=6*gap_hour)  # Large gap - will interpolate

    # Both should work but may give different results
    assert not result_small_gap.isnull().any().any()
    assert not result_large_gap.isnull().any().any()


def test_sd_measures_timezone_parameter():
    """Test the timezone parameter."""
    hours = 48
    dt0 = 5
    samples = int(hours * 60/dt0)
    dates = pd.date_range('2020-01-01', periods=samples, freq=f'{dt0}min')
    data = pd.DataFrame({
        'id': ['subject1'] * samples,
        'time': dates,
        'gl': np.random.normal(120, 20, samples)
    })

    # Test with different timezone
    result_utc = iglu.sd_measures(data, tz="UTC")
    result_no_tz = iglu.sd_measures(data)

    # Results should be the same regardless of timezone for this data
    for col in [ 'SDhhmm']:
        assert abs(result_utc.iloc[0][col] - result_no_tz.iloc[0][col]) < 1


def test_sd_measures_empty_dataframe():
    """Test that empty DataFrame raises appropriate error."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])

    with pytest.raises(ValueError):
        iglu.sd_measures(data)


def test_sd_measures_output_dtypes():
    """Test that output has correct data types."""
    dates = pd.date_range('2020-01-01', periods=48, freq='1h')
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
    dates = pd.date_range('2020-01-01', periods=48, freq='1h')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': np.random.normal(120, 20, 48)
    })

    result1 = iglu.sd_measures(data)
    result2 = iglu.sd_measures(data)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_sd_measures_series_with_datetime_index():
    """Test SD measures calculation with Series input that has DatetimeIndex."""
    # Create test data with DatetimeIndex
    time = pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00',
                          '2020-01-02 10:00:00', '2020-01-02 10:05:00', '2020-01-02 10:10:00'])
    data = pd.Series(
        [100, 120, 110,  # Day 1: mean=110, std=10
         90, 130, 95],   # Day 2: mean=105, std=21.21
        index=time
    )

    # Calculate SD measures
    result = iglu.sd_measures(data)

    # Expected results:
    # SDmean = mean([10, 21.21]) = 15.605
    # SDsd = std([10, 21.21], ddof=1) = 7.931
    expected = {
        'SDw': 15.8972,
        'SDhhmm':15.612495,
        'SDwsh': 16.341298,
        'SDdm': 3.535534,
        'SDb': 8.249579,
        'SDbdm': 7.071068
    }

    # Compare results
    assert isinstance(result, dict)
    assert len(result) == 6
    np.testing.assert_allclose(result['SDw'], expected['SDw'], rtol=0.001)
    np.testing.assert_allclose(result['SDhhmm'], expected['SDhhmm'], rtol=0.001)
    np.testing.assert_allclose(result['SDwsh'], expected['SDwsh'], rtol=0.001)
    np.testing.assert_allclose(result['SDdm'], expected['SDdm'], rtol=0.001)
    np.testing.assert_allclose(result['SDb'], expected['SDb'], rtol=0.001)
    np.testing.assert_allclose(result['SDbdm'], expected['SDbdm'], rtol=0.001)


def test_sd_measures_series_without_datetime_index():
    """Test SD measures calculation with Series input that doesn't have DatetimeIndex."""
    # Create test data with regular index
    data = pd.Series(
        [100, 120, 110, 90, 130, 95],
        index=range(6)  # Regular integer index instead of DatetimeIndex
    )

    # Attempt to calculate SD measures - should raise ValueError
    with pytest.raises(ValueError, match="Series must have a DatetimeIndex"):
        iglu.sd_measures(data)
