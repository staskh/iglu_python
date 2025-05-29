import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'range_glu'

def get_test_scenarios():
    """Get test scenarios for range_glu calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    # Filter scenarios for range_glu method
    return [scenario for scenario in expected_results if scenario['method'] == method_name]

@pytest.fixture
def test_data():
    """Fixture that provides test data for range_glu calculations"""
    return get_test_scenarios()

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_range_glu_calculation(scenario):
    """Test range_glu calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.range_glu(df, **kwargs)
    
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

def test_range_glu_output_format():
    """Test the output format of range_glu function"""
    
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
    result = iglu.range_glu(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'range'])
    
    # Check values are non-negative
    assert all(result['range'] >= 0)
    
    # Test with Series input
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result_series = iglu.range_glu(series_data)
    assert isinstance(result_series, pd.DataFrame)
    assert 'range' in result_series.columns
    assert len(result_series) == 1
    assert result_series['range'].iloc[0] == 25  # max(165) - min(140)
    
    # Test with empty data
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result_empty = iglu.range_glu(empty_data)
    assert isinstance(result_empty, pd.DataFrame)
    assert len(result_empty) == 0
    
    # Test with single subject and constant glucose
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 150, 150, 150]  # Constant glucose
    })
    result_single = iglu.range_glu(single_subject)
    assert len(result_single) == 1
    assert result_single['range'].iloc[0] == 0  # Should be 0 for constant glucose
    
    # Test with missing values
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, np.nan, 160, 165]
    })
    result_na = iglu.range_glu(data_with_na)
    assert isinstance(result_na, pd.DataFrame)
    assert len(result_na) == 1
    assert result_na['range'].iloc[0] == 15  # max(165) - min(150)
    
    # Test with multiple subjects and different ranges
    multi_subject = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                              '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [150, 200, 130, 190]
    })
    result_multi = iglu.range_glu(multi_subject)
    assert len(result_multi) == 2
    assert result_multi.loc[result_multi['id'] == 'subject1', 'range'].iloc[0] == 50  # 200 - 150
    assert result_multi.loc[result_multi['id'] == 'subject2', 'range'].iloc[0] == 60  # 190 - 130 