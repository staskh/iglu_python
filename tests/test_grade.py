import pytest
import pandas as pd
import numpy as np
from iglu_python.grade import grade, _grade_formula

def test_grade_formula():
    """Test the helper function that calculates GRADE scores for individual values."""
    # Test with perfect glucose value (should give low GRADE score)
    assert _grade_formula(np.array([100])) < 10
    
    # Test with high glucose value (should give high GRADE score)
    assert _grade_formula(np.array([300])) > 40
    
    # Test with very high glucose value (should be capped at 50)
    assert _grade_formula(np.array([500])) == 50
    
    # Test with multiple values
    values = np.array([100, 200, 300])
    scores = _grade_formula(values)
    assert len(scores) == 3
    assert all(scores >= 0) and all(scores <= 50)

def test_grade_basic():
    """Test basic GRADE calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [100, 200, 100, 100]  # One subject has better control
    })
    
    result = grade(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'GRADE' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Subject 1 has higher GRADE score due to higher glucose values
    assert result.loc[result['id'] == 'subject1', 'GRADE'].values[0] > \
           result.loc[result['id'] == 'subject2', 'GRADE'].values[0]

def test_grade_series_input():
    """Test GRADE calculation with Series input."""
    data = pd.Series([100, 200, 100, 100])
    result = grade(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'GRADE' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1

def test_grade_empty_data():
    """Test GRADE calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        grade(data)

def test_grade_missing_values():
    """Test GRADE calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, np.nan, 200]
    })
    
    result = grade(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'GRADE'])
    assert len(result) == 1

def test_grade_constant_values():
    """Test GRADE calculation with constant glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, 100, 100]  # All values are the same
    })
    
    result = grade(data)
    
    # Check that GRADE score is consistent
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, 'GRADE'] == _grade_formula(np.array([100]))[0]

def test_grade_multiple_subjects():
    """Test GRADE calculation with multiple subjects."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject3', 'subject3'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [100, 100, 200, 200, 100, 200]  # Different patterns for each subject
    })
    
    result = grade(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has best control (lowest GRADE)
    assert result.loc[result['id'] == 'subject1', 'GRADE'].values[0] < \
           result.loc[result['id'] == 'subject2', 'GRADE'].values[0]
    # Subject 2 has worst control (highest GRADE)
    assert result.loc[result['id'] == 'subject2', 'GRADE'].values[0] > \
           result.loc[result['id'] == 'subject3', 'GRADE'].values[0]
    # Subject 3 has mixed control (middle GRADE)
    assert result.loc[result['id'] == 'subject3', 'GRADE'].values[0] > \
           result.loc[result['id'] == 'subject1', 'GRADE'].values[0] 