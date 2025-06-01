import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'modd'

def get_test_scenarios():
    """Get test scenarios for modd calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)
    # Filter scenarios for modd method
    return [scenario for scenario in expected_results['test_runs'] if scenario['method'] == method_name]

@pytest.fixture
def test_data():
    """Fixture that provides test data for modd calculations"""
    return get_test_scenarios()

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_modd_iglu_r_compatible(scenario):
    """Test modd calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.modd(df, **kwargs)
    
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

def test_modd_default_output():
    """Test modd calculation with default parameters"""
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
    
    result = iglu.modd(data)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'MODD'])
    assert all(result['MODD'] >= 0)

def test_modd_custom_lag():
    """Test modd calculation with custom lag value"""
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
    
    result = iglu.modd(data, lag=2)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'MODD'])
    assert all(result['MODD'] >= 0)

def test_modd_series_input():
    """Test modd calculation with Series input"""
    series_data = pd.Series([150, 200, 180, 160, 140, 190])
    result = iglu.modd(series_data)
    assert isinstance(result, pd.DataFrame)
    assert 'MODD' in result.columns
    assert len(result) == 1

def test_modd_empty_input():
    """Test modd calculation with empty DataFrame"""
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result = iglu.modd(empty_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_modd_single_subject():
    """Test modd calculation with single subject data"""
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 150, 150, 150]  # Constant glucose
    })
    result = iglu.modd(single_subject)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'MODD'])
    assert len(result) == 1
    """Test the output format of modd function"""
    
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
    
    # Test with default parameters
    result = iglu.modd(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'MODD'])
    
    # Check values are non-negative
    assert all(result['MODD'] >= 0)
    
    # Test with different lag values
    result_lag2 = iglu.modd(data, lag=2)
    assert isinstance(result_lag2, pd.DataFrame)
    assert all(col in result_lag2.columns for col in ['id', 'MODD'])
    assert all(result_lag2['MODD'] >= 0)
    
    # Test with Series input
    series_data = pd.Series([150, 200, 180, 160, 140, 190])
    result_series = iglu.modd(series_data)
    assert isinstance(result_series, pd.DataFrame)
    assert 'MODD' in result_series.columns
    assert len(result_series) == 1
    
    # Test with empty data
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result_empty = iglu.modd(empty_data)
    assert isinstance(result_empty, pd.DataFrame)
    assert len(result_empty) == 0
    
    # Test with single subject and constant glucose
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 150, 150, 150]  # Constant glucose
    })
    result_single = iglu.modd(single_subject)
    assert len(result_single) == 1
    assert result_single['MODD'].iloc[0] == 0  # Should be 0 for constant glucose
    
    # Test with missing values
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, np.nan, 180, 160]
    })
    result_na = iglu.modd(data_with_na)
    assert isinstance(result_na, pd.DataFrame)
    assert len(result_na) == 1
    
    # Test with timezone parameter
    result_tz = iglu.modd(data, tz='UTC')
    assert len(result_tz) == 1
    assert isinstance(result_tz['MODD'].iloc[0], float)
    
    # Test with multiple days of data
    multi_day_data = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-02 00:00:00', '2020-01-02 00:05:00',
            '2020-01-02 00:10:00', '2020-01-02 00:15:00'
        ]),
        'gl': [150, 200, 180, 160, 140, 190, 170, 210]
    })
    result_multi = iglu.modd(multi_day_data)
    assert len(result_multi) == 1
    assert isinstance(result_multi['MODD'].iloc[0], float)
    
    # Test with insufficient data points
    small_data = pd.DataFrame({
        'id': ['subject1'] * 3,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [150, 160, 170]
    })
    result_small = iglu.modd(small_data)
    assert len(result_small) == 1
    assert isinstance(result_small['MODD'].iloc[0], float)
    
    # Test with lag larger than available data
    result_large_lag = iglu.modd(multi_day_data, lag=3)
    assert len(result_large_lag) == 1
    assert pd.isna(result_large_lag['MODD'].iloc[0])  # Should be NaN for insufficient data 