import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'mag'

def get_test_scenarios():
    """Get test scenarios for MAG calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    # Filter scenarios for MAG method
    return [scenario for scenario in expected_results if scenario['method'] == method_name]

@pytest.fixture
def test_data():
    """Fixture that provides test data for MAG calculations"""
    return get_test_scenarios()

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_mag_calculation(scenario):
    """Test MAG calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.mag(df, **kwargs)
    
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

def test_mag_output_format():
    """Test the output format of mag function"""
    
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
    result = iglu.mag(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'MAG'])
    
    # Check values are non-negative
    assert all(result['MAG'] >= 0)
    
    # Test with Series input
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result_series = iglu.mag(series_data)
    assert isinstance(result_series, pd.DataFrame)
    assert 'MAG' in result_series.columns
    assert len(result_series) == 1
    
    # Test with empty data
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result_empty = iglu.mag(empty_data)
    assert isinstance(result_empty, pd.DataFrame)
    assert len(result_empty) == 0
    
    # Test with single subject and constant glucose
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 150, 150, 150]  # Constant glucose
    })
    result_single = iglu.mag(single_subject)
    assert len(result_single) == 1
    assert result_single['MAG'].iloc[0] == 0  # Should be 0 for constant glucose
    
    # Test with missing values
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, np.nan, 160, 165]
    })
    result_na = iglu.mag(data_with_na)
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
    
    # Test with n=30 (30 minutes)
    result_n30 = iglu.mag(data_multi, n=30)
    assert len(result_n30) == 1
    
    # Test with n=60 (1 hour)
    result_n60 = iglu.mag(data_multi, n=60)
    assert len(result_n60) == 1
    assert result_n60['MAG'].iloc[0] != result_n30['MAG'].iloc[0]  # Different n should give different results
    
    # Test with n less than data collection frequency
    result_n2 = iglu.mag(data_multi, n=2)  # n=2 is less than default dt0=5
    assert len(result_n2) == 1
    assert result_n2['MAG'].iloc[0] > 0  # Should still work, using dt0 as minimum
    
    # Test with timezone parameter
    result_tz = iglu.mag(data_multi, tz='UTC')
    assert len(result_tz) == 1
    assert isinstance(result_tz['MAG'].iloc[0], float)
    
    # Test with custom dt0
    result_dt0 = iglu.mag(data_multi, dt0=10)  # 10 minutes between measurements
    assert len(result_dt0) == 1
    assert isinstance(result_dt0['MAG'].iloc[0], float)
    
    # Test with custom inter_gap
    result_gap = iglu.mag(data_multi, inter_gap=60)  # 60 minutes max gap
    assert len(result_gap) == 1
    assert isinstance(result_gap['MAG'].iloc[0], float) 