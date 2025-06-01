import pytest
import pandas as pd
import numpy as np
import json
import iglu_python as iglu
from iglu_python.sd_roc import sd_roc

method_name = 'sd_roc'

def get_test_scenarios():
    """Get test scenarios for SD ROC calculations"""
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    # Filter scenarios for SD ROC method
    return [scenario for scenario in expected_results['test_runs'] if scenario['method'] == method_name]

@pytest.mark.parametrize('scenario', get_test_scenarios())
def test_sd_roc_iglu_r_compatible(scenario):
    """Test SD ROC calculation against expected results"""
    
    input_file_name = scenario['input_file_name']
    kwargs = scenario['kwargs']
    
    expected_results = scenario['results']
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    expected_df = expected_df.dropna(subset=['sd_roc'])
    if expected_df.empty:
        pytest.skip("This SD ROC test has no numeric value to compare")

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.sd_roc(df, **kwargs)
    
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

def test_sd_roc_basic():
    """Test basic SD of ROC calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [100, 120, 100, 80]  # Different rates of change for each subject
    })
    
    result = sd_roc(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'sd_roc' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Both subjects have same absolute rate of change (4 mg/dL per minute)
    assert abs(result.loc[result['id'] == 'subject1', 'sd_roc'].values[0] - \
               result.loc[result['id'] == 'subject2', 'sd_roc'].values[0]) < 1e-10

def test_sd_roc_series_input():
    """Test SD of ROC calculation with Series input."""
    data = pd.Series([100, 120, 100, 80],
                     index=pd.to_datetime(['2020-01-01 00:00:00',
                                         '2020-01-01 00:05:00',
                                         '2020-01-01 00:10:00',
                                         '2020-01-01 00:15:00']))
    result = sd_roc(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'sd_roc' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1
    
    # Check that SD of ROC is calculated
    assert not np.isnan(result.loc[0, 'sd_roc'])
    assert result.loc[0, 'sd_roc'] > 0

def test_sd_roc_series_input_no_datetime_index():
    """Test SD of ROC calculation with Series input without datetime index."""
    data = pd.Series([100, 120, 100, 80])
    with pytest.raises(ValueError):
        sd_roc(data)

def test_sd_roc_empty_data():
    """Test SD of ROC calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        sd_roc(data)

def test_sd_roc_missing_values():
    """Test SD of ROC calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, np.nan, 80]
    })
    
    result = sd_roc(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'sd_roc'])
    assert len(result) == 1

def test_sd_roc_single_value():
    """Test SD of ROC calculation with only one value per subject."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:00:00']),
        'gl': [100, 120]
    })
    
    result = sd_roc(data)
    
    # Check that NaN is returned for single values
    assert isinstance(result, pd.DataFrame)
    assert np.isnan(result.loc[0, 'sd_roc'])
    assert np.isnan(result.loc[1, 'sd_roc'])

def test_sd_roc_constant_values():
    """Test SD of ROC calculation with constant glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, 100, 100]  # Constant glucose values
    })
    
    result = sd_roc(data)
    
    # Check that SD of ROC is 0 for constant values
    assert abs(result.loc[0, 'sd_roc']) < 1e-10

def test_sd_roc_multiple_subjects():
    """Test SD of ROC calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1',
               'subject2', 'subject2', 'subject2',
               'subject3', 'subject3', 'subject3'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, 100, 100,  # Subject 1: constant
               100, 120, 140,  # Subject 2: linear increase
               100, 140, 100]  # Subject 3: high variability
    })
    
    result = sd_roc(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has lowest SD of ROC (constant values)
    assert result.loc[result['id'] == 'subject1', 'sd_roc'].values[0] <= \
           result.loc[result['id'] == 'subject2', 'sd_roc'].values[0]
    # Subject 3 has highest SD of ROC (high variability)
    assert result.loc[result['id'] == 'subject3', 'sd_roc'].values[0] > \
           result.loc[result['id'] == 'subject2', 'sd_roc'].values[0]

def test_sd_roc_irregular_timestamps():
    """Test SD of ROC calculation with irregular time intervals."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:15:00', '2020-01-01 00:20:00']),
        'gl': [100, 120, 140, 160]  # Regular glucose increase with irregular time intervals
    })
    
    result = sd_roc(data)
    
    # Check that SD of ROC is calculated correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'sd_roc'])
    # SD of ROC should be positive but not extremely high
    assert 0 < result.loc[0, 'sd_roc'] < 100 