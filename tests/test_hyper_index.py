import pytest
import pandas as pd
import numpy as np
from iglu_python.hyper_index import hyper_index

def test_hyper_index_basic():
    """Test basic hyper_index calculation with known values."""
    # Create test data with known glucose values
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2', 'subject2'],
        'time': pd.date_range(start='2020-01-01', periods=6, freq='5min'),
        'gl': [150, 200, 180, 130, 190, 160]
    })
    
    # Calculate hyper_index
    result = hyper_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'hyper_index' in result.columns
    assert len(result) == 2  # Two subjects
    
    # Check that hyper_index values are non-negative
    assert all(result['hyper_index'] >= 0)
    
    # Check that subject2 has lower hyper_index than subject1
    # (since subject1 has more values above ULTR)
    subject1_index = result[result['id'] == 'subject1']['hyper_index'].iloc[0]
    subject2_index = result[result['id'] == 'subject2']['hyper_index'].iloc[0]
    assert subject1_index > subject2_index

def test_hyper_index_series_input():
    """Test hyper_index calculation with Series input."""
    # Create test data as Series
    data = pd.Series([150, 200, 180, 130, 190, 160])
    
    # Calculate hyper_index
    result = hyper_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'hyper_index' in result.columns
    assert 'id' not in result.columns
    assert len(result) == 1
    
    # Check that hyper_index value is non-negative
    assert result['hyper_index'].iloc[0] >= 0

def test_hyper_index_custom_parameters():
    """Test hyper_index calculation with custom parameters."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.date_range(start='2020-01-01', periods=3, freq='5min'),
        'gl': [150, 200, 180]
    })
    
    # Test with custom parameters
    result = hyper_index(data, ULTR=160, a=1.5, c=25)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'hyper_index' in result.columns
    assert len(result) == 1
    
    # Check that hyper_index value is non-negative
    assert result['hyper_index'].iloc[0] >= 0

def test_hyper_index_empty_data():
    """Test hyper_index calculation with empty data."""
    # Create empty DataFrame
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    
    # Test that function raises appropriate error
    with pytest.raises(ValueError):
        hyper_index(data)

def test_hyper_index_missing_values():
    """Test hyper_index calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.date_range(start='2020-01-01', periods=3, freq='5min'),
        'gl': [150, np.nan, 180]
    })
    
    # Calculate hyper_index
    result = hyper_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'hyper_index' in result.columns
    assert len(result) == 1
    
    # Check that hyper_index value is non-negative
    assert result['hyper_index'].iloc[0] >= 0

def test_hyper_index_no_hyper_values():
    """Test hyper_index calculation with no values above ULTR."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.date_range(start='2020-01-01', periods=3, freq='5min'),
        'gl': [130, 135, 138]  # All values below ULTR=140
    })
    
    # Calculate hyper_index
    result = hyper_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'hyper_index' in result.columns
    assert len(result) == 1
    
    # Check that hyper_index value is 0 (no values above ULTR)
    assert result['hyper_index'].iloc[0] == 0

def test_hyper_index_multiple_subjects():
    """Test hyper_index calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject3', 'subject3'],
        'time': pd.date_range(start='2020-01-01', periods=6, freq='5min'),
        'gl': [150, 200, 130, 190, 140, 140]
    })
    
    # Calculate hyper_index
    result = hyper_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'hyper_index' in result.columns
    assert len(result) == 3  # Three subjects
    
    # Check that hyper_index values are non-negative
    assert all(result['hyper_index'] >= 0)
    
    # Check that subject3 has lower hyper_index than others (since values are at ULTR)
    subject3_index = result[result['id'] == 'subject3']['hyper_index'].iloc[0]
    subject1_index = result[result['id'] == 'subject1']['hyper_index'].iloc[0]
    subject2_index = result[result['id'] == 'subject2']['hyper_index'].iloc[0]
    assert subject3_index <= subject1_index
    assert subject3_index <= subject2_index 