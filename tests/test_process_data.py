import json
import warnings

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "process_data"


def get_test_scenarios():
    """Get test scenarios for process_data calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for process_data method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_process_data_iglu_r_compatible(scenario):
    """Test process_data calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df['time'] = expected_df['time'].apply(lambda x: pd.to_datetime(x).tz_convert('UTC'))
    expected_df = expected_df.reset_index(drop=True)

    result_df = iglu.process_data(df, **kwargs)

    assert result_df is not None

    result_df = result_df.reset_index(drop=True)
    result_df['time'] = result_df['time'].dt.tz_convert('UTC')

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


def test_process_data_basic():
    """Test basic process_data functionality."""
    data = pd.DataFrame({
        'subject_id': ['A', 'A', 'B', 'B'],
        'datetime': ['2020-01-01 10:00:00', '2020-01-01 10:05:00', 
                     '2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'glucose': [120, 130, 110, 125]
    })
    
    result = iglu.process_data(data, id='subject_id', timestamp='datetime', glu='glucose')
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['id', 'time', 'gl']
    assert len(result) == 4
    
    # Check data types
    assert pd.api.types.is_string_dtype(result['id'])
    for col in ['gl']:
        assert np.issubdtype(result[col].dtype, np.number)
    
    # Check values
    assert set(result['id'].unique()) == {'A', 'B'}
    assert all(result['gl'] >= 0)


def test_process_data_default_id():
    """Test process_data with default id assignment."""
    data = pd.DataFrame({
        'datetime': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'glucose': [120, 130]
    })
    
    # Capture the print output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    result = iglu.process_data(data, timestamp='datetime', glu='glucose')
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Check that the default id message was printed
    assert "No 'id' parameter passed, defaulting id to 1" in captured_output.getvalue()
    
    # Check result
    assert list(result.columns) == ['id', 'time', 'gl']
    assert all(result['id'] == '1')


def test_process_data_case_insensitive():
    """Test that column matching is case insensitive."""
    data = pd.DataFrame({
        'Subject_ID': ['A', 'A'],
        'DateTime': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'Glucose': [120, 130]
    })
    
    result = iglu.process_data(data, id='subject_id', timestamp='datetime', glu='glucose')
    
    assert list(result.columns) == ['id', 'time', 'gl']
    assert len(result) == 2


def test_process_data_mmol_conversion():
    """Test mmol/L to mg/dL conversion."""
    data = pd.DataFrame({
        'id': ['A', 'A'],
        'time': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'glucose_mmol/l': [6.7, 7.2]  # mmol/L values
    })
    
    result = iglu.process_data(data, id='id', timestamp='time', glu='glucose_mmol/l')
    
    # Check conversion (6.7 mmol/L * 18 = 120.6 mg/dL)
    assert abs(result['gl'].iloc[0] - 120.6) < 0.1
    assert abs(result['gl'].iloc[1] - 129.6) < 0.1


def test_process_data_series_with_datetime_index():
    """Test process_data with Series input having datetime index."""
    dates = pd.date_range('2020-01-01 10:00:00', periods=3, freq='5min')
    series_data = pd.Series([120, 130, 125], index=dates)
    
    result = iglu.process_data(series_data)
    
    assert list(result.columns) == ['id', 'time', 'gl']
    assert len(result) == 3
    assert all(result['id'] == '1')


def test_process_data_series_without_datetime_index():
    """Test process_data with Series input without datetime index."""
    series_data = pd.Series([120, 130, 125])

    with pytest.raises(ValueError):
        iglu.process_data(series_data)


def test_process_data_list_input():
    """Test process_data with list input."""
    glucose_list = [120, 130, 125, 140]
    
    with pytest.raises(ValueError):
        iglu.process_data(glucose_list)


def test_process_data_numpy_array():
    """Test process_data with numpy array input."""
    glucose_array = np.array([120, 130, 125, 140])
    
    with pytest.raises(ValueError):
        iglu.process_data(glucose_array)
    


def test_process_data_missing_values():
    """Test handling of missing values."""
    data = pd.DataFrame({
        'id': ['A', 'A', 'A', 'A'],
        'time': ['2020-01-01 10:00:00', '2020-01-01 10:05:00', 
                 '2020-01-01 10:10:00', '2020-01-01 10:15:00'],
        'gl': [120, np.nan, 125, 140]
    })
    
    result = iglu.process_data(data, id='id', timestamp='time', glu='gl')
    
    # Should remove rows with NaN glucose values
    assert len(result) == 3
    assert not result['gl'].isna().any()


def test_process_data_glucose_warnings():
    """Test glucose value validation warnings."""
    data = pd.DataFrame({
        'id': ['A', 'A', 'A'],
        'time': ['2020-01-01 10:00:00', '2020-01-01 10:05:00', '2020-01-01 10:10:00'],
        'gl': [10, 120, 600]  # Very low and very high values
    })
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = iglu.process_data(data, id='id', timestamp='time', glu='gl')
        
        # Should generate warnings for extreme values
        warning_messages = [str(warning.message) for warning in w]
        assert any("below 20" in msg for msg in warning_messages)
        assert any("above 500" in msg for msg in warning_messages)


def test_process_data_column_rename():
    """Test for column name renaming."""
    data = pd.DataFrame({
        'subject': ['A', 'A'],
        'datetime': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'glucose': [120, 130]
    })

    result = iglu.process_data(data, id='subject', timestamp='datetime', glu='glucose')

    # Check column names and order
    assert list(result.columns) == ['id', 'time', 'gl']
    
    # Check data types
    assert pd.api.types.is_string_dtype(result['id'])
    assert pd.api.types.is_datetime64_any_dtype(result['time'])
    assert pd.api.types.is_numeric_dtype(result['gl'])
    
    # Check values were preserved
    assert all(result['id'] == 'A')
    assert result['gl'].tolist() == [120, 130]


def test_process_data_column_not_found_errors():
    """Test error handling for missing columns."""
    data = pd.DataFrame({
        'subject': ['A', 'A'],
        'datetime': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'glucose': [120, 130]
    })
    
    # Test missing id column
    with pytest.raises(ValueError, match="Column 'wrong_id' not found"):
        iglu.process_data(data, id='wrong_id', timestamp='datetime', glu='glucose')
    
    # Test missing timestamp column
    with pytest.raises(ValueError, match="Column 'wrong_time' not found"):
        iglu.process_data(data, id='subject', timestamp='wrong_time', glu='glucose')
    
    # Test missing glucose column
    with pytest.raises(ValueError, match="Column 'wrong_glucose' not found"):
        iglu.process_data(data, id='subject', timestamp='datetime', glu='wrong_glucose')


def test_process_data_alternative_column_suggestion():
    """Test error messages that suggest alternative columns."""
    data = pd.DataFrame({
        'id': ['A', 'A'],
        'time': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'gl': [120, 130]
    })
    
    # Test suggestion for id column
    with pytest.raises(ValueError, match="Fix user-defined argument name for id"):
        iglu.process_data(data, id='wrong_id', timestamp='time', glu='gl')


def test_process_data_invalid_data_types():
    """Test error handling for invalid data types."""
    # Test invalid data type
    with pytest.raises(TypeError, match="Invalid data type"):
        iglu.process_data("invalid_data")
    
    # Test invalid parameter types
    data = pd.DataFrame({
        'id': ['A', 'A'],
        'time': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'gl': [120, 130]
    })
    
    with pytest.raises(ValueError, match="User-defined id name must be string"):
        iglu.process_data(data, id=123, timestamp='time', glu='gl')


def test_process_data_custom_time_parser():
    """Test custom time parser functionality."""
    data = pd.DataFrame({
        'id': ['A', 'A'],
        'time': ['01/01/2020 10:00:00', '01/01/2020 10:05:00'],
        'gl': [120, 130]
    })
    
    # Custom parser for MM/DD/YYYY format
    custom_parser = lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M:%S')
    
    result = iglu.process_data(data, id='id', timestamp='time', glu='gl', 
                              time_parser=custom_parser)
    
    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(result['time'])


def test_process_data_empty_after_processing():
    """Test error when no data remains after processing."""
    data = pd.DataFrame({
        'id': ['A', 'A'],
        'time': ['2020-01-01 10:00:00', '2020-01-01 10:05:00'],
        'gl': [np.nan, np.nan]  # All NaN glucose values
    })
    
    with pytest.raises(ValueError):
        iglu.process_data(data, id='id', timestamp='time', glu='gl')


def test_process_data_empty_input():
    """Test error handling for empty input."""
    empty_data = pd.DataFrame()
    
    with pytest.raises(ValueError, match="No data remaining after removing NAs"):
        iglu.process_data(empty_data)


def test_process_data_list_with_column_specs_error():
    """Test error when providing column specs with list input."""
    glucose_list = [120, 130, 125]
    
    with pytest.raises(ValueError, match="Cannot process list/array data with column specifications"):
        iglu.process_data(glucose_list, id='id', timestamp='time', glu='gl') 


def test_process_data_output_dtypes():
    """Test that output has correct data types."""
    dates = pd.date_range('2020-01-01', periods=48, freq='1H')
    data = pd.DataFrame({
        'id': ['subject1'] * 48,
        'time': dates,
        'gl': np.random.normal(120, 20, 48)
    })
    
    result = iglu.process_data(data, id='id', timestamp='time', glu='gl')
    
    # Check data types
    assert pd.api.types.is_string_dtype(result['id'])
    assert pd.api.types.is_datetime64_any_dtype(result['time'])
    assert pd.api.types.is_numeric_dtype(result['gl']) 