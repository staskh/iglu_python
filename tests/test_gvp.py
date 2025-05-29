import pytest
import pandas as pd
import numpy as np
from iglu_python.gvp import gvp

def test_gvp_basic():
    """Test basic GVP calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [100, 120, 100, 80]  # Different rates of change for each subject
    })
    
    result = gvp(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'GVP' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Both subjects have same absolute rate of change (4 mg/dL per minute)
    assert abs(result.loc[result['id'] == 'subject1', 'GVP'].values[0] - \
               result.loc[result['id'] == 'subject2', 'GVP'].values[0]) < 1e-10

def test_gvp_series_input():
    """Test GVP calculation with Series input."""
    data = pd.Series([100, 120, 100, 80],
                     index=pd.to_datetime(['2020-01-01 00:00:00',
                                         '2020-01-01 00:05:00',
                                         '2020-01-01 00:10:00',
                                         '2020-01-01 00:15:00']))
    result = gvp(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'GVP' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1
    
    # Check that GVP is calculated
    assert not np.isnan(result.loc[0, 'GVP'])
    assert result.loc[0, 'GVP'] > 0

def test_gvp_series_input_no_datetime_index():
    """Test GVP calculation with Series input without datetime index."""
    data = pd.Series([100, 120, 100, 80])
    with pytest.raises(ValueError):
        gvp(data)

def test_gvp_empty_data():
    """Test GVP calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        gvp(data)

def test_gvp_missing_values():
    """Test GVP calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, np.nan, 80]
    })
    
    result = gvp(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'GVP'])
    assert len(result) == 1

def test_gvp_single_value():
    """Test GVP calculation with only one value per subject."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:00:00']),
        'gl': [100, 120]
    })
    
    result = gvp(data)
    
    # Check that NaN is returned for single values
    assert isinstance(result, pd.DataFrame)
    assert np.isnan(result.loc[0, 'GVP'])
    assert np.isnan(result.loc[1, 'GVP'])

def test_gvp_constant_values():
    """Test GVP calculation with constant glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, 100, 100]  # Constant glucose values
    })
    
    result = gvp(data)
    
    # Check that GVP is 0 for constant values
    assert abs(result.loc[0, 'GVP']) < 1e-10

def test_gvp_multiple_subjects():
    """Test GVP calculation with multiple subjects."""
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
    
    result = gvp(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has lowest GVP (constant values)
    assert result.loc[result['id'] == 'subject1', 'GVP'].values[0] < \
           result.loc[result['id'] == 'subject2', 'GVP'].values[0]
    # Subject 3 has highest GVP (high variability)
    assert result.loc[result['id'] == 'subject3', 'GVP'].values[0] > \
           result.loc[result['id'] == 'subject2', 'GVP'].values[0]

def test_gvp_irregular_timestamps():
    """Test GVP calculation with irregular time intervals."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:15:00', '2020-01-01 00:20:00']),
        'gl': [100, 120, 140, 160]  # Regular glucose increase with irregular time intervals
    })
    
    result = gvp(data)
    
    # Check that GVP is calculated correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'GVP'])
    # GVP should be positive but not extremely high
    assert 0 < result.loc[0, 'GVP'] < 100
 