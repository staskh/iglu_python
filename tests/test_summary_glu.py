import json

import numpy as np
import pandas as pd
import pytest

import iglu_python as iglu

method_name = "summary_glu"


def get_test_scenarios():
    """Get test scenarios for summary_glu calculations"""
    # Load expected results
    with open("tests/expected_results.json", "r") as f:
        expected_results = json.load(f)
    # set local timezone
    iglu.utils.set_local_tz(expected_results["config"]["local_tz"])
    # Filter scenarios for summary_glu method
    return [
        scenario
        for scenario in expected_results["test_runs"]
        if scenario["method"] == method_name
    ]


@pytest.mark.parametrize("scenario", get_test_scenarios())
def test_summary_glu_iglu_r_compatible(scenario):
    """Test summary_glu calculation against expected results"""

    input_file_name = scenario["input_file_name"]
    kwargs = scenario["kwargs"]

    # Read CSV and convert time column to datetime
    df = pd.read_csv(input_file_name, index_col=0)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    expected_results = scenario["results"]
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    result_df = iglu.summary_glu(df, **kwargs)

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


def test_summary_glu_basic_dataframe():
    """Test basic summary_glu functionality with DataFrame."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2', 'subject2'],
        'time': pd.date_range(start='2020-01-01', periods=6, freq='5min'),
        'gl': [150, 200, 180, 130, 190, 160]
    })

    result = iglu.summary_glu(data)

    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two subjects
    
    # Check columns
    expected_columns = ['id', 'Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
    assert list(result.columns) == expected_columns
    
    # Check data types
    assert pd.api.types.is_string_dtype(result['id'])
    for col in ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']:
        assert pd.api.types.is_numeric_dtype(result[col])
    
    # Check that we have the correct subjects
    assert set(result['id']) == {'subject1', 'subject2'}
    
    # Check that summary values are reasonable
    subject1_data = result[result['id'] == 'subject1'].iloc[0]
    assert subject1_data['Min.'] <= subject1_data['Max.']
    assert subject1_data['1st Qu.'] <= subject1_data['Median'] <= subject1_data['3rd Qu.']


def test_summary_glu_single_subject():
    """Test summary_glu with single subject."""
    data = pd.DataFrame({
        'id': ['subject1'] * 5,
        'time': pd.date_range(start='2020-01-01', periods=5, freq='5min'),
        'gl': [100, 120, 140, 160, 180]
    })

    result = iglu.summary_glu(data)

    assert len(result) == 1
    assert result.iloc[0]['id'] == 'subject1'
    
    # Check specific values for known data
    row = result.iloc[0]
    assert row['Min.'] == 100
    assert row['Max.'] == 180
    assert row['Median'] == 140  # Middle value
    assert row['Mean'] == 140  # Average of 100,120,140,160,180
    assert row['1st Qu.'] == 120  # 25th percentile
    assert row['3rd Qu.'] == 160  # 75th percentile


def test_summary_glu_vector_input_series():
    """Test summary_glu with Series input."""
    glucose_series = pd.Series([100, 120, 140, 160, 180])
    
    result = iglu.summary_glu(glucose_series)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    
    # Should not have id column for vector input
    expected_columns = ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
    assert list(result.columns) == expected_columns
    
    # Check values
    row = result.iloc[0]
    assert row['Min.'] == 100
    assert row['Max.'] == 180
    assert row['Median'] == 140
    assert row['Mean'] == 140


def test_summary_glu_vector_input_list():
    """Test summary_glu with list input."""
    glucose_list = [100, 120, 140, 160, 180]
    
    result = iglu.summary_glu(glucose_list)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    
    # Should not have id column for vector input
    expected_columns = ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
    assert list(result.columns) == expected_columns


def test_summary_glu_vector_input_numpy():
    """Test summary_glu with numpy array input."""
    glucose_array = np.array([100, 120, 140, 160, 180])
    
    result = iglu.summary_glu(glucose_array)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    
    # Should not have id column for vector input
    expected_columns = ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
    assert list(result.columns) == expected_columns


def test_summary_glu_missing_values():
    """Test handling of missing glucose values."""
    data = pd.DataFrame({
        'id': ['subject1'] * 6,
        'time': pd.date_range(start='2020-01-01', periods=6, freq='5min'),
        'gl': [100, np.nan, 140, 160, np.nan, 180]
    })

    result = iglu.summary_glu(data)

    assert len(result) == 1
    # Should calculate stats only on non-NaN values: [100, 140, 160, 180]
    row = result.iloc[0]
    assert row['Min.'] == 100
    assert row['Max.'] == 180
    assert row['Mean'] == 145  # (100+140+160+180)/4


def test_summary_glu_missing_values_vector():
    """Test handling of missing values in vector input."""
    glucose_series = pd.Series([100, np.nan, 140, 160, np.nan, 180])
    
    result = iglu.summary_glu(glucose_series)
    
    assert len(result) == 1
    # Should calculate stats only on non-NaN values: [100, 140, 160, 180]
    row = result.iloc[0]
    assert row['Min.'] == 100
    assert row['Max.'] == 180
    assert row['Mean'] == 145


def test_summary_glu_all_missing_values():
    """Test behavior with all missing glucose values for a subject."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.date_range(start='2020-01-01', periods=4, freq='5min'),
        'gl': [np.nan, np.nan, 100, 120]
    })

    with pytest.warns(UserWarning, match="No valid glucose values found for subject subject1"):
        result = iglu.summary_glu(data)

    assert len(result) == 2
    
    # subject1 should have all NaN values
    subject1_row = result[result['id'] == 'subject1'].iloc[0]
    for col in ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']:
        assert pd.isna(subject1_row[col])
    
    # subject2 should have valid values
    subject2_row = result[result['id'] == 'subject2'].iloc[0]
    assert not pd.isna(subject2_row['Mean'])


def test_summary_glu_all_missing_vector():
    """Test error with all missing values in vector input."""
    glucose_series = pd.Series([np.nan, np.nan, np.nan])
    
    with pytest.raises(ValueError, match="No valid glucose values found"):
        iglu.summary_glu(glucose_series)


def test_summary_glu_single_value():
    """Test summary_glu with single glucose value."""
    data = pd.DataFrame({
        'id': ['subject1'],
        'time': pd.date_range(start='2020-01-01', periods=1, freq='5min'),
        'gl': [150]
    })

    result = iglu.summary_glu(data)

    assert len(result) == 1
    row = result.iloc[0]
    
    # All summary stats should be the same for single value
    for col in ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']:
        assert row[col] == 150


def test_summary_glu_multiple_subjects():
    """Test summary_glu with multiple subjects having different characteristics."""
    data = pd.DataFrame({
        'id': ['A'] * 3 + ['B'] * 4 + ['C'] * 2,
        'time': pd.date_range(start='2020-01-01', periods=9, freq='5min'),
        'gl': [100, 110, 120,  # Subject A: low glucose
               200, 210, 220, 230,  # Subject B: high glucose  
               150, 160]  # Subject C: medium glucose
    })

    result = iglu.summary_glu(data)

    assert len(result) == 3
    assert set(result['id']) == {'A', 'B', 'C'}
    
    # Check that B has higher values than A
    a_mean = result[result['id'] == 'A']['Mean'].iloc[0]
    b_mean = result[result['id'] == 'B']['Mean'].iloc[0]
    c_mean = result[result['id'] == 'C']['Mean'].iloc[0]
    
    assert a_mean < c_mean < b_mean


def test_summary_glu_identical_values():
    """Test summary_glu with identical glucose values."""
    data = pd.DataFrame({
        'id': ['subject1'] * 5,
        'time': pd.date_range(start='2020-01-01', periods=5, freq='5min'),
        'gl': [120] * 5  # All identical values
    })

    result = iglu.summary_glu(data)

    assert len(result) == 1
    row = result.iloc[0]
    
    # All summary stats should be the same for identical values
    for col in ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']:
        assert row[col] == 120


def test_summary_glu_empty_dataframe():
    """Test error handling for empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    
    with pytest.raises(ValueError):
        iglu.summary_glu(data)


def test_summary_glu_column_order():
    """Test that output columns are in the correct order."""
    data = pd.DataFrame({
        'id': ['subject1'] * 3,
        'time': pd.date_range(start='2020-01-01', periods=3, freq='5min'),
        'gl': [100, 150, 200]
    })

    result = iglu.summary_glu(data)
    
    expected_columns = ['id', 'Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']
    assert list(result.columns) == expected_columns


def test_summary_glu_percentile_accuracy():
    """Test that percentiles are calculated correctly."""
    # Use a specific dataset to verify percentile calculations
    glucose_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data = pd.DataFrame({
        'id': ['subject1'] * 10,
        'time': pd.date_range(start='2020-01-01', periods=10, freq='5min'),
        'gl': glucose_values
    })

    result = iglu.summary_glu(data)
    row = result.iloc[0]
    
    # Verify against numpy percentile calculations
    assert row['Min.'] == np.min(glucose_values)
    assert row['1st Qu.'] == np.percentile(glucose_values, 25)
    assert row['Median'] == np.median(glucose_values)
    assert row['Mean'] == np.mean(glucose_values)
    assert row['3rd Qu.'] == np.percentile(glucose_values, 75)
    assert row['Max.'] == np.max(glucose_values) 