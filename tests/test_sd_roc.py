import pytest
import pandas as pd
import numpy as np
from iglu_python.sd_roc import sd_roc

def test_sd_roc_basic():
    """Test basic SD of ROC calculation with known glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:00:00', '2020-01-01 00:05:00']),
        'gl': [100, 120, 100, 80]  # Different rates of change for each subject
    })
    
    result = sd_roc(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'SD_ROC' in result.columns
    assert len(result) == 2
    
    # Check calculations
    # Both subjects have same absolute rate of change (4 mg/dL per minute)
    assert abs(result.loc[result['id'] == 'subject1', 'SD_ROC'].values[0] - \
               result.loc[result['id'] == 'subject2', 'SD_ROC'].values[0]) < 1e-10

def test_sd_roc_series_input():
    """Test SD of ROC calculation with Series input."""
    data = pd.Series([100, 120, 100, 80],
                     index=pd.to_datetime(['2020-01-01 00:00:00',
                                         '2020-01-01 00:05:00',
                                         '2020-01-01 00:10:00',
                                         '2020-01-01 00:15:00']))
    result = sd_roc(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'SD_ROC' in result.columns
    assert len(result) == 1
    assert len(result.columns) == 1
    
    # Check that SD of ROC is calculated
    assert not np.isnan(result.loc[0, 'SD_ROC'])
    assert result.loc[0, 'SD_ROC'] > 0

def test_sd_roc_series_input_no_datetime_index():
    """Test SD of ROC calculation with Series input without datetime index."""
    data = pd.Series([100, 120, 100, 80])
    with pytest.raises(ValueError):
        sd_roc(data)

def test_sd_roc_empty_data():
    """Test SD of ROC calculation with empty DataFrame."""
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    with pytest.raises(ValueError):
        sd_roc(data)

def test_sd_roc_missing_values():
    """Test SD of ROC calculation with missing values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, np.nan, 80]
    })
    
    result = sd_roc(data)
    
    # Check that NaN values are handled correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'SD_ROC'])
    assert len(result) == 1

def test_sd_roc_single_value():
    """Test SD of ROC calculation with only one value per subject."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject2'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:00:00']),
        'gl': [100, 120]
    })
    
    result = sd_roc(data)
    
    # Check that NaN is returned for single values
    assert isinstance(result, pd.DataFrame)
    assert np.isnan(result.loc[0, 'SD_ROC'])
    assert np.isnan(result.loc[1, 'SD_ROC'])

def test_sd_roc_constant_values():
    """Test SD of ROC calculation with constant glucose values."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00']),
        'gl': [100, 100, 100]  # Constant glucose values
    })
    
    result = sd_roc(data)
    
    # Check that SD of ROC is 0 for constant values
    assert abs(result.loc[0, 'SD_ROC']) < 1e-10

def test_sd_roc_multiple_subjects():
    """Test SD of ROC calculation with multiple subjects."""
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
    
    result = sd_roc(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check relative values
    # Subject 1 has lowest SD of ROC (constant values)
    assert result.loc[result['id'] == 'subject1', 'SD_ROC'].values[0] < \
           result.loc[result['id'] == 'subject2', 'SD_ROC'].values[0]
    # Subject 3 has highest SD of ROC (high variability)
    assert result.loc[result['id'] == 'subject3', 'SD_ROC'].values[0] > \
           result.loc[result['id'] == 'subject2', 'SD_ROC'].values[0]

def test_sd_roc_irregular_timestamps():
    """Test SD of ROC calculation with irregular time intervals."""
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:15:00', '2020-01-01 00:20:00']),
        'gl': [100, 120, 140, 160]  # Regular glucose increase with irregular time intervals
    })
    
    result = sd_roc(data)
    
    # Check that SD of ROC is calculated correctly
    assert isinstance(result, pd.DataFrame)
    assert not np.isnan(result.loc[0, 'SD_ROC'])
    # SD of ROC should be positive but not extremely high
    assert 0 < result.loc[0, 'SD_ROC'] < 100 