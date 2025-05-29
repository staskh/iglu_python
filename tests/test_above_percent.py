import pytest
import pandas as pd
import json
import iglu_python as iglu

method_name = 'above_percent'

@pytest.fixture
def test_data():
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    method_scenarios = [scenario for scenario in expected_results if scenario['method'] == method_name]

    for scenario in method_scenarios:
        yield scenario

def test_above_percent_calculation(test_data):
    """Test above_percent calculation against expected results"""
    
    input_file_name = test_data['input_file_name']
    kwargs = test_data['kwargs']
    
    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    result_df = iglu.above_percent(df, **kwargs)
    
    assert result_df is not None
    
    expected_results = test_data['results']
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)
    
    # Compare DataFrames with precision to 0.001
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

def test_above_percent_output_format():
    """Test the output format of above_percent function"""
    
    # Create test data
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', 
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [150, 200, 130, 190]
    })
    
    # Test with default targets
    result = iglu.above_percent(data)
    
    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert all(col.startswith('above_') for col in result.columns if col != 'id')
    
    # Check values are between 0 and 100
    for col in result.columns:
        if col != 'id':
            assert all((result[col] >= 0) & (result[col] <= 100))
    
    # Test with custom targets
    custom_targets = [150, 200]
    result_custom = iglu.above_percent(data, targets_above=custom_targets)
    
    # Check custom target columns
    assert all(f'above_{t}' in result_custom.columns for t in custom_targets)
    
    # Test with Series input
    result_series = iglu.above_percent(data['gl'], targets_above=custom_targets)
    assert 'id' not in result_series.columns
    assert len(result_series) == 1  # Single row for Series input 