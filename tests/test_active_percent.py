import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'active_percent'

def get_test_scenarios():
    """Get test scenarios for active_percent calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)
    # set local timezone    
    iglu.utils.set_local_tz(expected_results['config']['local_tz'])
    # Filter scenarios for active_percent method
    return [scenario for scenario in expected_results['test_runs'] if scenario['method'] == method_name]

@pytest.fixture
def test_data():
    """Fixture that provides test data for active_percent calculations"""
    return get_test_scenarios()

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_active_percent_iglu_r_compatible(scenario):
    """Test active_percent calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.active_percent(df, **kwargs)
    
    assert result_df is not None
    
    expected_results = scenario['results']
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    # Convert start_date and end_date to Timestamp
    for col in ['start_date', 'end_date']:
        if col in expected_df.columns:
            expected_df[col] = pd.to_datetime(expected_df[col])
    
    # Convert timestamp columns to strings for comparison
    for col in ['start_date', 'end_date']:
        if col in result_df.columns:
            # drop tz information
            result_df[col] = result_df[col].dt.tz_localize(None)
    
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
        rtol=0.01
    )

def test_active_percent_output_format():
    """Test the output format of active_percent function"""
    
    # Create test data with known gaps
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',  # 0 min
            '2020-01-01 00:05:00',  # 5 min (gap)
            '2020-01-01 00:15:00',  # 15 min
            '2020-01-01 00:20:00',  # 20 min
            '2020-01-01 00:00:00',  # subject2
            '2020-01-01 00:05:00'   # subject2
        ]),
        'gl': [150, np.nan, 160, 165, 140, 145]
    })
    
    # Test with default parameters
    result = iglu.active_percent(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'active_percent', 'ndays', 'start_date', 'end_date'])
    
    # Check values are between 0 and 100
    assert all((result['active_percent'] >= 0) & (result['active_percent'] <= 100))
    
    # Check ndays is non-negative
    assert all(result['ndays'] >= 0)
    
    # Test with custom dt0
    result_custom = iglu.active_percent(data, dt0=5)
    assert isinstance(result_custom, pd.DataFrame)
    
    # Test with consistent end date
    end_date = datetime(2020, 1, 1, 1, 0)  # 1 hour after start
    result_consistent = iglu.active_percent(data, consistent_end_date=end_date)
    assert all(result_consistent['end_date'] == end_date)
    
    # Test with timezone
    result_tz = iglu.active_percent(data, tz='GMT')
    assert isinstance(result_tz, pd.DataFrame)
    
    # Test with empty data
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        iglu.active_percent(empty_data)
    
    # Test with single subject and no gaps
    single_subject = pd.DataFrame({
        'id': ['subject1'] * 3,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [150, 155, 160]
    })
    result_single = iglu.active_percent(single_subject, dt0=5)
    assert len(result_single) == 1
    assert result_single['active_percent'].iloc[0] == 100.0  # Should be 100% active with no gaps 