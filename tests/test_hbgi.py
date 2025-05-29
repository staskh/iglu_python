import pytest
import pandas as pd
import numpy as np
from iglu_python.hbgi import hbgi

def test_hbgi_basic():
    """Test basic HBGI calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [150, 200, 130, 190]  # Different hyperglycemia for each subject
    })
    
    result = hbgi(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'HBGI' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Subject 1 has higher hyperglycemia (higher glucose values)
    assert result.loc[result['id'] == 'subject1', 'HBGI'].values[0] > \
           result.loc[result['id'] == 'subject2', 'HBGI'].values[0]

def test_hbgi_series_input():
    """Test HBGI calculation with Series input."""
    data = pd.Series([150, 200, 130, 190])
    result = hbgi(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'HBGI' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1
    
    # Check that HBGI is calculated (should be positive due to hyperglycemia)
    assert result.loc[0, 'HBGI'] > 0

def test_hbgi_empty_data():
    """Test HBGI calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        hbgi(data)

def test_hbgi_missing_values():
    """Test HBGI calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [150, np.nan, 200]
    })
    
    result = hbgi(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'HBGI'])
    assert len(result) == 1

def test_hbgi_all_below_threshold():
    """Test HBGI calculation when all values are below threshold."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [80, 90, 100]  # All values below 112.5
    })
    
    result = hbgi(data)
    
    # Check that HBGI is 0 when all values are below threshold
    assert abs(result.loc[0, 'HBGI']) < 1e-10

def test_hbgi_all_above_threshold():
    """Test HBGI calculation when all values are above threshold."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [200, 250, 300]  # All values above 112.5
    })
    
    result = hbgi(data)
    
    # Check that HBGI is positive when all values are above threshold
    assert result.loc[0, 'HBGI'] > 0

def test_hbgi_multiple_subjects():
    """Test HBGI calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject3', 'subject3'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [80, 80, 200, 200, 80, 200]  # Different patterns for each subject
    })
    
    result = hbgi(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has lowest HBGI (all values below threshold)
    assert result.loc[result['id'] == 'subject1', 'HBGI'].values[0] < \
           result.loc[result['id'] == 'subject2', 'HBGI'].values[0]
    # Subject 2 has highest HBGI (all values above threshold)
    assert result.loc[result['id'] == 'subject2', 'HBGI'].values[0] > \
           result.loc[result['id'] == 'subject3', 'HBGI'].values[0]
    # Subject 3 has middle HBGI (mixed values)
    assert result.loc[result['id'] == 'subject3', 'HBGI'].values[0] > \
           result.loc[result['id'] == 'subject1', 'HBGI'].values[0]

def test_hbgi_edge_cases():
    """Test HBGI calculation with edge case glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [112.4, 112.5, 112.6, 500]  # Values around and above threshold
    })
    
    result = hbgi(data)
    
    # Check that HBGI is calculated correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'HBGI'])
    # HBGI should be positive but not extremely high
    assert 0 < result.loc[0, 'HBGI'] < 100 