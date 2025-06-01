import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'auc'

def get_test_scenarios():
    """Get test scenarios for AUC calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results['config']['local_tz'])
    # Filter scenarios for AUC method
    return [scenario for scenario in expected_results['test_runs'] if scenario['method'] == method_name]


@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_auc_iglu_r_compatible(scenario):
    """Test AUC calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    expected_results = scenario['results']
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    result_df = iglu.auc(df, **kwargs)
    
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

def test_auc_output_format():
    """Test the output format of auc function"""
    
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
    result = iglu.auc(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'hourly_auc'])
    
    # Check values are non-negative
    assert all(result['hourly_auc'] >= 0)
    
    # # Test with timezone
    # result_tz = iglu.auc(data, tz='GMT')
    # assert isinstance(result_tz, pd.DataFrame)
    
    # # Test with empty data
    # empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    # result_empty = iglu.auc(empty_data)
    # assert isinstance(result_empty, pd.DataFrame)
    # assert len(result_empty) == 0
    
    # # Test with single subject and constant glucose
    # single_subject = pd.DataFrame({
    #     'id': ['subject1'] * 4,
    #     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
    #                           '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
    #     'gl': [150, 150, 150, 150]  # Constant glucose
    # })
    # result_single = iglu.auc(single_subject)
    # assert len(result_single) == 1
    # assert result_single['hourly_auc'].iloc[0] == 150.0  # Should be equal to constant glucose
    
    # # Test with missing values
    # data_with_na = pd.DataFrame({
    #     'id': ['subject1'] * 4,
    #     'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
    #                           '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
    #     'gl': [150, np.nan, 160, 165]
    # })
    # result_na = iglu.auc(data_with_na)
    # assert isinstance(result_na, pd.DataFrame)
    # assert len(result_na) == 1
    # assert not np.isnan(result_na['hourly_auc'].iloc[0])  # Should handle NA values
    
    # # Test with multiple days
    # multi_day = pd.DataFrame({
    #     'id': ['subject1'] * 8,
    #     'time': pd.to_datetime([
    #         '2020-01-01 00:00:00', '2020-01-01 00:05:00',  # Day 1
    #         '2020-01-01 00:10:00', '2020-01-01 00:15:00',
    #         '2020-01-02 00:00:00', '2020-01-02 00:05:00',  # Day 2
    #         '2020-01-02 00:10:00', '2020-01-02 00:15:00'
    #     ]),
    #     'gl': [150, 155, 160, 165, 140, 145, 150, 155]
    # })
    # result_multi = iglu.auc(multi_day)
    # assert isinstance(result_multi, pd.DataFrame)
    # assert len(result_multi) == 1
    # assert not np.isnan(result_multi['hourly_auc'].iloc[0])  # Should handle multiple days 