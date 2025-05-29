import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'conga'

def get_test_scenarios():
    """Get test scenarios for CONGA calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    # Filter scenarios for CONGA method
    return [scenario for scenario in expected_results if scenario['method'] == method_name]

@pytest.fixture
def test_data():
    """Fixture that provides test data for CONGA calculations"""
    return get_test_scenarios()

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_conga_calculation(scenario):
    """Test CONGA calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.conga(df, **kwargs)
    
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

def test_conga_output_format():
    """Test the output format of conga function"""
    
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
        'gl': [150, 155, 160, 165, 140, 145]
    })
    
    # Test with default parameters
    result = iglu.conga(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'CONGA'])
    
    # Check values are non-negative
    assert all(result['CONGA'] >= 0)
    
    # Test with Series input
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result_series = iglu.conga(series_data)
    assert isinstance(result_series, pd.DataFrame)
    assert 'CONGA' in result_series.columns
    assert len(result_series) == 1
    
    # Test with empty data
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result_empty = iglu.conga(empty_data)
    assert isinstance(result_empty, pd.DataFrame)
    assert len(result_empty) == 0
    
    # Test with single subject and constant glucose
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 150, 150, 150]  # Constant glucose
    })
    result_single = iglu.conga(single_subject)
    assert len(result_single) == 1
    assert result_single['CONGA'].iloc[0] == 0  # Should be 0 for constant glucose
    
    # Test with missing values
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, np.nan, 160, 165]
    })
    result_na = iglu.conga(data_with_na)
    assert isinstance(result_na, pd.DataFrame)
    assert len(result_na) == 1
    
    # Test with different n values
    data_multi = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-01 00:20:00', '2020-01-01 00:25:00',
            '2020-01-01 00:30:00', '2020-01-01 00:35:00'
        ]),
        'gl': [150, 160, 170, 180, 190, 200, 210, 220]
    })
    
    # Test with n=1 (1 hour)
    result_n1 = iglu.conga(data_multi, n=1)
    assert len(result_n1) == 1
    
    # Test with n=2 (2 hours)
    result_n2 = iglu.conga(data_multi, n=2)
    assert len(result_n2) == 1
    assert result_n2['CONGA'].iloc[0] != result_n1['CONGA'].iloc[0]  # Different n should give different results
    
    # Test with timezone parameter
    result_tz = iglu.conga(data_multi, tz='UTC')
    assert len(result_tz) == 1
    assert isinstance(result_tz['CONGA'].iloc[0], float) 