import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import iglu_python as iglu

method_name = 'episode_calculation'

def get_test_scenarios():
    """Get test scenarios for episode calculation"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    # Filter scenarios for episode_calculation method
    return [scenario for scenario in expected_results['test_runs'] if scenario['method'] == method_name]

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_episode_iglu_r_compatible(scenario):
    """Test episode calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    expected_results = scenario['results']
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.episode_iglu_r_compatible(df, **kwargs)
    
    assert result_df is not None
    
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
        rtol=1e-3,
    )

def test_episode_calculation_default():
    """Test episode calculation with default parameters"""
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
    
    result = iglu.episode_iglu_r_compatible(data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'type', 'level', 'avg_ep_per_day', 
                                               'avg_ep_duration', 'avg_ep_gl', 'total_episodes'])
    assert len(result) > 0  # Should have at least one row per subject

def test_episode_calculation_series():
    """Test episode calculation with Series input"""
    series_data = pd.Series([150, 155, 160, 165, 140, 145])
    result = iglu.episode_iglu_r_compatible(series_data)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['id', 'type', 'level', 'avg_ep_per_day', 
                                               'avg_ep_duration', 'avg_ep_gl', 'total_episodes'])
    assert len(result) > 0

def test_episode_calculation_empty():
    """Test episode calculation with empty data"""
    empty_data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        result = iglu.episode_iglu_r_compatible(empty_data)

def test_episode_calculation_constant_glucose():
    """Test episode calculation with constant glucose values"""
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:15:00'
        ]),
        'gl': [150, 150, 150, 150]
    })
    result = iglu.episode_iglu_r_compatible(data)
    assert len(result) > 0
    # No episodes should be detected for constant glucose
    assert all(result['total_episodes'] == 0)

def test_episode_calculation_missing_values():
    """Test episode calculation with missing values"""
    data_with_na = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                              '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, np.nan, 160, 165]
    })
    result = iglu.episode_iglu_r_compatible(data_with_na)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

def test_episode_calculation_different_thresholds():
    """Test episode calculation with different thresholds"""
    data = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-01 00:20:00', '2020-01-01 00:25:00',
            '2020-01-01 00:30:00', '2020-01-01 00:35:00'
        ]),
        'gl': [150, 160, 170, 180, 190, 200, 210, 220]
    })
    
    result_default = iglu.episode_iglu_r_compatible(data)
    result_custom = iglu.episode_iglu_r_compatible(data, lv1_hypo=100, lv2_hypo=80,
                                           lv1_hyper=200, lv2_hyper=250)
    assert len(result_default) > 0
    assert len(result_custom) > 0
    # Different thresholds should give different results
    assert not result_default.equals(result_custom)

def test_episode_calculation_return_data():
    """Test episode calculation with return_data=True"""
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:15:00'
        ]),
        'gl': [150, 160, 170, 180]
    })
    result = iglu.episode_iglu_r_compatible(data, return_data=True)
    assert isinstance(result, dict)
    assert 'episodes' in result
    assert 'data' in result
    assert isinstance(result['episodes'], pd.DataFrame)
    assert isinstance(result['data'], pd.DataFrame)
    assert all(col in result['data'].columns for col in ['id', 'time', 'gl', 'lv1_hypo',
                                                        'lv2_hypo', 'lv1_hyper', 'lv2_hyper',
                                                        'ext_hypo', 'lv1_hypo_excl', 'lv1_hyper_excl'])

def test_episode_calculation_extended_hypo():
    """Test episode calculation with extended hypoglycemia"""
    data = pd.DataFrame({
        'id': ['subject1'] * 24,  # 2 hours of data
        'time': pd.date_range(start='2020-01-01 00:00:00', periods=24, freq='5min'),
        'gl': [65] * 24  # 2 hours below lv1_hypo
    })
    result = iglu.episode_iglu_r_compatible(data)
    assert len(result) > 0
    # Should detect extended hypoglycemia
    extended_hypo = result[(result['type'] == 'hypo') & (result['level'] == 'extended')]
    assert len(extended_hypo) > 0
    assert extended_hypo['total_episodes'].iloc[0] > 0

def test_episode_calculation_exclusive_levels():
    """Test episode calculation with exclusive level 1 events"""
    data = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-01 00:20:00', '2020-01-01 00:25:00',
            '2020-01-01 00:30:00', '2020-01-01 00:35:00'
        ]),
        'gl': [60, 65, 70, 75, 50, 55, 60, 65]  # Mix of lv1 and lv2 hypo
    })
    result = iglu.episode_iglu_r_compatible(data)
    assert len(result) > 0
    # Should detect exclusive level 1 events
    lv1_excl = result[(result['type'] == 'hypo') & (result['level'] == 'lv1_excl')]
    assert len(lv1_excl) > 0
    assert lv1_excl['total_episodes'].iloc[0] > 0 