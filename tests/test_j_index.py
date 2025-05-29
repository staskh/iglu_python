import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import iglu_python as iglu

def test_j_index_calculation():
    """Test basic functionality of j_index"""
    
    # Create test data with known values
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject2', 'subject2'],
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00'
        ]),
        'gl': [150, 200, 130, 190]
    })
    
    # Test with DataFrame input
    result = iglu.j_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert 'id' in result.columns
    assert 'J_index' in result.columns
    assert len(result) == 2  # Two subjects
    
    # Check calculations
    # For subject1: mean = 175, sd = 25, J-index = 0.001 * (175 + 25)^2 = 1.5625
    # For subject2: mean = 160, sd = 30, J-index = 0.001 * (160 + 30)^2 = 1.4400
    assert abs(result.loc[result['id'] == 'subject1', 'J_index'].iloc[0] - 1.5625) < 0.0001
    assert abs(result.loc[result['id'] == 'subject2', 'J_index'].iloc[0] - 1.4400) < 0.0001
    
    # Test with Series input
    result_series = iglu.j_index(data['gl'])
    
    # Check output format for Series input
    assert isinstance(result_series, pd.DataFrame)
    assert 'J_index' in result_series.columns
    assert 'id' not in result_series.columns
    assert len(result_series) == 1
    
    # Check calculation for Series input
    # Overall mean = 167.5, sd = 27.5, J-index = 0.001 * (167.5 + 27.5)^2 = 1.5000
    assert abs(result_series['J_index'].iloc[0] - 1.5000) < 0.0001

def test_j_index_empty_data():
    """Test j_index with empty data"""
    
    # Empty DataFrame
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    result = iglu.j_index(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert 'id' in result.columns
    assert 'J_index' in result.columns
    
    # Empty Series
    data_series = pd.Series(dtype=float)
    result_series = iglu.j_index(data_series)
    assert isinstance(result_series, pd.DataFrame)
    assert len(result_series) == 1
    assert np.isnan(result_series['J_index'].iloc[0])

def test_j_index_missing_values():
    """Test j_index with missing values"""
    
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00'
        ]),
        'gl': [150, np.nan, 200]
    })
    
    result = iglu.j_index(data)
    
    # Check that missing values are handled appropriately
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert not np.isnan(result['J_index'].iloc[0])
    
    # For subject1: mean = 175, sd = 25, J-index = 0.001 * (175 + 25)^2 = 1.5625
    assert abs(result['J_index'].iloc[0] - 1.5625) < 0.0001

def test_j_index_constant_values():
    """Test j_index with constant glucose values"""
    
    data = pd.DataFrame({
        'id': ['subject1', 'subject1', 'subject1'],
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00'
        ]),
        'gl': [150, 150, 150]
    })
    
    result = iglu.j_index(data)
    
    # For constant values: mean = 150, sd = 0, J-index = 0.001 * (150 + 0)^2 = 0.225
    assert abs(result['J_index'].iloc[0] - 0.225) < 0.0001

def test_j_index_single_measurement():
    """Test j_index with single measurement per subject"""
    
    data = pd.DataFrame({
        'id': ['subject1', 'subject2'],
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:00:00'
        ]),
        'gl': [150, 200]
    })
    
    result = iglu.j_index(data)
    
    # For single measurements: sd = 0, J-index = 0.001 * (mean)^2
    assert abs(result.loc[result['id'] == 'subject1', 'J_index'].iloc[0] - 0.225) < 0.0001  # 0.001 * 150^2
    assert abs(result.loc[result['id'] == 'subject2', 'J_index'].iloc[0] - 0.400) < 0.0001  # 0.001 * 200^2

def test_j_index_multiple_subjects():
    """Test j_index with multiple subjects and varying data points"""
    
    data = pd.DataFrame({
        'id': ['subject1'] * 3 + ['subject2'] * 2 + ['subject3'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00',  # subject1
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',                        # subject2
            '2020-01-01 00:00:00', '2020-01-01 00:05:00', '2020-01-01 00:10:00', '2020-01-01 00:15:00'  # subject3
        ]),
        'gl': [150, 200, 180,  # subject1
               130, 190,       # subject2
               140, 160, 170, 180]  # subject3
    })
    
    result = iglu.j_index(data)
    
    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Three subjects
    assert set(result['id']) == {'subject1', 'subject2', 'subject3'}
    
    # Check calculations for each subject
    # subject1: mean = 176.67, sd = 20.82, J-index = 0.001 * (176.67 + 20.82)^2 = 1.5625
    # subject2: mean = 160, sd = 30, J-index = 0.001 * (160 + 30)^2 = 1.4400
    # subject3: mean = 162.5, sd = 14.43, J-index = 0.001 * (162.5 + 14.43)^2 = 1.5625
    assert abs(result.loc[result['id'] == 'subject1', 'J_index'].iloc[0] - 1.5625) < 0.0001
    assert abs(result.loc[result['id'] == 'subject2', 'J_index'].iloc[0] - 1.4400) < 0.0001
    assert abs(result.loc[result['id'] == 'subject3', 'J_index'].iloc[0] - 1.5625) < 0.0001 