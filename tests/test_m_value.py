import pytest
import pandas as pd
import numpy as np
from iglu_python.m_value import m_value

def test_m_value_basic():
    """Test basic M-value calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [90, 180, 90, 90]  # One subject has perfect control, other has high values
    })
    
    result = m_value(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'M_value' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Subject 1 has perfect control (all values at reference), should have M-value close to 0
    assert result.loc[result['id'] == 'subject1', 'M_value'].values[0] < 1
    # Subject 2 has high values, should have higher M-value
    assert result.loc[result['id'] == 'subject2', 'M_value'].values[0] > 100

def test_m_value_series_input():
    """Test M-value calculation with Series input."""
    data = pd.Series([90, 180, 90, 90])
    result = m_value(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'M_value' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1

def test_m_value_custom_reference():
    """Test M-value calculation with custom reference value."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [100, 200]
    })
    
    # Test with different reference values
    result1 = m_value(data, r=100)
    result2 = m_value(data, r=200)
    
    # M-value should be lower with higher reference value for these data
    assert result1.loc[0, 'M_value'] > result2.loc[0, 'M_value']

def test_m_value_empty_data():
    """Test M-value calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        m_value(data)

def test_m_value_missing_values():
    """Test M-value calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [90, np.nan, 180]
    })
    
    result = m_value(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'M_value'])
    assert len(result) == 1

def test_m_value_constant_values():
    """Test M-value calculation with constant glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [90, 90, 90]  # All values at reference
    })
    
    result = m_value(data)
    
    # M-value should be very close to 0 for perfect control
    assert result.loc[0, 'M_value'] < 1

def test_m_value_multiple_subjects():
    """Test M-value calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject3', 'subject3'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [90, 90, 180, 180, 90, 180]  # Different patterns for each subject
    })
    
    result = m_value(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has perfect control
    assert result.loc[result['id'] == 'subject1', 'M_value'].values[0] < 1
    # Subject 2 has high values
    assert result.loc[result['id'] == 'subject2', 'M_value'].values[0] > 100
    # Subject 3 has mixed values
    m_value3 = result.loc[result['id'] == 'subject3', 'M_value'].values[0]
    assert 1 < m_value3 < 100 