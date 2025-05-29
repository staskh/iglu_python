import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'mage'

def get_test_scenarios():
    """Get test scenarios for MAGE calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    # Filter scenarios for MAGE method
    return [scenario for scenario in expected_results if scenario['method'] == method_name]

@pytest.fixture
def test_data():
    """Fixture that provides test data for MAGE calculations"""
    return get_test_scenarios()

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_mage_calculation(scenario):
    """Test MAGE calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.mage(df, **kwargs)
    
    assert result_df is not None
    
    expected_results = scenario['results']
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

def test_mage_output_format():
    """Test the output format of mage function"""
    
    # Create test data with known values
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',  # 0 min
            '2020-01-01 00:05:00',  # 5 min
            '2020-01-01 00:10:00',  # 10 min
            '2020-01-01 00:15:00',  # 15 min
            '2020-01-01 00:00:00',  # subject2
            '2020-01-01 00:05:00'   # subject2
        ]),
        'gl': [150, 200, 180, 160, 140, 190]
    })
    
    # Test with default parameters (ma version)
    result = iglu.mage(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'MAGE'])
    
    # Check values are non-negative
    assert all(result['MAGE'] >= 0)
    
    # Test with naive version
    result_naive = iglu.mage(data, version='naive')
    assert isinstance(result_naive, pd.DataFrame)
    assert all(col in result_naive.columns for col in ['id', 'MAGE'])
    assert all(result_naive['MAGE'] >= 0)
    
    # Test with Series input
    series_data = pd.Series([150, 200, 180, 160, 140, 190])
    result_series = iglu.mage(series_data)
    assert isinstance(result_series, pd.DataFrame)
    assert 'MAGE' in result_series.columns
    assert len(result_series) == 1
    
    # Test with empty data
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result_empty = iglu.mage(empty_data)
    assert isinstance(result_empty, pd.DataFrame)
    assert len(result_empty) == 0
    
    # Test with single subject and constant glucose
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 150, 150, 150]  # Constant glucose
    })
    result_single = iglu.mage(single_subject)
    assert len(result_single) == 1
    assert pd.isna(result_single['MAGE'].iloc[0])  # Should be NaN for constant glucose
    
    # Test with missing values
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, np.nan, 180, 160]
    })
    result_na = iglu.mage(data_with_na)
    assert isinstance(result_na, pd.DataFrame)
    assert len(result_na) == 1
    
    # Test with different directions
    data_multi = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-01 00:20:00', '2020-01-01 00:25:00',
            '2020-01-01 00:30:00', '2020-01-01 00:35:00'
        ]),
        'gl': [150, 200, 180, 160, 140, 190, 170, 210]
    })
    
    # Test with direction='plus'
    result_plus = iglu.mage(data_multi, direction='plus')
    assert len(result_plus) == 1
    assert isinstance(result_plus['MAGE'].iloc[0], float)
    
    # Test with direction='minus'
    result_minus = iglu.mage(data_multi, direction='minus')
    assert len(result_minus) == 1
    assert isinstance(result_minus['MAGE'].iloc[0], float)
    
    # Test with direction='max'
    result_max = iglu.mage(data_multi, direction='max')
    assert len(result_max) == 1
    assert isinstance(result_max['MAGE'].iloc[0], float)
    
    # Test with different moving average windows
    result_ma = iglu.mage(data_multi, short_ma=3, long_ma=6)
    assert len(result_ma) == 1
    assert isinstance(result_ma['MAGE'].iloc[0], float)
    
    # Test with return_type='df'
    result_df = iglu.mage(data_multi, return_type='df')
    assert isinstance(result_df, pd.DataFrame)
    assert all(col in result_df.columns for col in ['id', 'start', 'end', 'mage', 'plus_or_minus', 'first_excursion'])
    
    # Test with timezone parameter
    result_tz = iglu.mage(data_multi, tz='UTC')
    assert len(result_tz) == 1
    assert isinstance(result_tz['MAGE'].iloc[0], float)
    
    # Test with custom inter_gap
    result_gap = iglu.mage(data_multi, inter_gap=60)
    assert len(result_gap) == 1
    assert isinstance(result_gap['MAGE'].iloc[0], float)
    
    # Test with custom max_gap
    result_max_gap = iglu.mage(data_multi, max_gap=120)
    assert len(result_max_gap) == 1
    assert isinstance(result_max_gap['MAGE'].iloc[0], float)
    
    # Test with sd_multiplier for naive version
    result_sd = iglu.mage(data_multi, version='naive', sd_multiplier=1.5)
    assert len(result_sd) == 1
    assert isinstance(result_sd['MAGE'].iloc[0], float)
    
    # Test that swapping short_ma and long_ma works correctly
    result_swapped = iglu.mage(data_multi, short_ma=32, long_ma=5)  # Should be automatically swapped
    assert len(result_swapped) == 1
    assert isinstance(result_swapped['MAGE'].iloc[0], float)
    
    # Test with insufficient data points
    small_data = pd.DataFrame({
        'id': ['subject1'] * 3,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [150, 160, 170]
    })
    result_small = iglu.mage(small_data)
    assert len(result_small) == 1
    assert pd.isna(result_small['MAGE'].iloc[0])  # Should be NaN for insufficient data 