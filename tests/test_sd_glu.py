import pytest
import pandas as pd
import numpy as np
from iglu_python.sd_glu import sd_glu

def test_sd_glu_basic():
    """Test basic sd_glu calculation with known values."""
    # Create test data with known glucose values
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2', 'subject2'],
        'time': pd.date_range(start='2020-01-01', periods=6, freq='5min'),
        'gl': [150, 200, 180, 130, 190, 160]
    })
    
    # Calculate sd_glu
    result = sd_glu(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'SD' in result.columns
    assert len(result) == 2  # Two subjects
    
    # Check that SD values are non-negative
    assert all(result['SD'] >= 0)
    
    # Check that subject2 has higher SD than subject1
    # (since subject2 has more variability in values)
    subject1_sd = result[result['id'] == 'subject1']['SD'].iloc[0]
    subject2_sd = result[result['id'] == 'subject2']['SD'].iloc[0]
    assert subject2_sd > subject1_sd

def test_sd_glu_series_input():
    """Test sd_glu calculation with Series input."""
    # Create test data as Series
    data = pd.Series([150, 200, 180, 130, 190, 160])
    
    # Calculate sd_glu
    result = sd_glu(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'SD' in result.columns
    assert 'id' not in result.columns
    assert len(result) == 1
    
    # Check that SD value is non-negative
    assert result['SD'].iloc[0] >= 0

def test_sd_glu_empty_data():
    """Test sd_glu calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    
    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        sd_glu(data)

def test_sd_glu_missing_values():
    """Test sd_glu calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.date_range(start='2020-01-01', periods=3, freq='5min'),
        'gl': [150, np.nan, 180]
    })
    
    # Calculate sd_glu
    result = sd_glu(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'SD' in result.columns
    assert len(result) == 1
    
    # Check that SD value is non-negative
    assert result['SD'].iloc[0] >= 0

def test_sd_glu_constant_values():
    """Test sd_glu calculation with constant values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.date_range(start='2020-01-01', periods=3, freq='5min'),
        'gl': [150, 150, 150]  # All values are the same
    })
    
    # Calculate sd_glu
    result = sd_glu(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'SD' in result.columns
    assert len(result) == 1
    
    # Check that SD value is 0 for constant values
    assert result['SD'].iloc[0] == 0

def test_sd_glu_multiple_subjects():
    """Test sd_glu calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject3', 'subject3'],
        'time': pd.date_range(start='2020-01-01', periods=6, freq='5min'),
        'gl': [150, 200, 130, 190, 140, 140]
    })
    
    # Calculate sd_glu
    result = sd_glu(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'SD' in result.columns
    assert len(result) == 3  # Three subjects
    
    # Check that SD values are non-negative
    assert all(result['SD'] >= 0)
    
    # Check that subject3 has lower SD than others (since values are more stable)
    subject3_sd = result[result['id'] == 'subject3']['SD'].iloc[0]
    subject1_sd = result[result['id'] == 'subject1']['SD'].iloc[0]
    subject2_sd = result[result['id'] == 'subject2']['SD'].iloc[0]
    assert subject3_sd <= subject1_sd
    assert subject3_sd <= subject2_sd 