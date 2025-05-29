import pytest
import pandas as pd
import numpy as np
from iglu_python.lbgi import lbgi

def test_lbgi_basic():
    """Test basic LBGI calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [70, 80, 60, 50]  # Different hypoglycemia for each subject
    })
    
    result = lbgi(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'LBGI' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Subject 2 has higher hypoglycemia (lower glucose values)
    assert result.loc[result['id'] == 'subject2', 'LBGI'].values[0] > \
           result.loc[result['id'] == 'subject1', 'LBGI'].values[0]

def test_lbgi_series_input():
    """Test LBGI calculation with Series input."""
    data = pd.Series([70, 80, 60, 50])
    result = lbgi(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'LBGI' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1
    
    # Check that LBGI is calculated (should be positive due to hypoglycemia)
    assert result.loc[0, 'LBGI'] > 0

def test_lbgi_empty_data():
    """Test LBGI calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        lbgi(data)

def test_lbgi_missing_values():
    """Test LBGI calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [70, np.nan, 60]
    })
    
    result = lbgi(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'LBGI'])
    assert len(result) == 1

def test_lbgi_all_above_threshold():
    """Test LBGI calculation when all values are above threshold."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [120, 130, 140]  # All values above 112.5
    })
    
    result = lbgi(data)
    
    # Check that LBGI is 0 when all values are above threshold
    assert abs(result.loc[0, 'LBGI']) < 1e-10

def test_lbgi_all_below_threshold():
    """Test LBGI calculation when all values are below threshold."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [40, 50, 60]  # All values below 112.5
    })
    
    result = lbgi(data)
    
    # Check that LBGI is positive when all values are below threshold
    assert result.loc[0, 'LBGI'] > 0

def test_lbgi_multiple_subjects():
    """Test LBGI calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject3', 'subject3'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [120, 120, 40, 40, 120, 40]  # Different patterns for each subject
    })
    
    result = lbgi(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has lowest LBGI (all values above threshold)
    assert result.loc[result['id'] == 'subject1', 'LBGI'].values[0] < \
           result.loc[result['id'] == 'subject2', 'LBGI'].values[0]
    # Subject 2 has highest LBGI (all values below threshold)
    assert result.loc[result['id'] == 'subject2', 'LBGI'].values[0] > \
           result.loc[result['id'] == 'subject3', 'LBGI'].values[0]
    # Subject 3 has middle LBGI (mixed values)
    assert result.loc[result['id'] == 'subject3', 'LBGI'].values[0] > \
           result.loc[result['id'] == 'subject1', 'LBGI'].values[0]

def test_lbgi_edge_cases():
    """Test LBGI calculation with edge case glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [112.4, 112.5, 112.6, 20]  # Values around and below threshold
    })
    
    result = lbgi(data)
    
    # Check that LBGI is calculated correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'LBGI'])
    # LBGI should be positive but not extremely high
    assert 0 < result.loc[0, 'LBGI'] < 100 