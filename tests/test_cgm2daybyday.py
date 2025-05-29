import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import iglu_python as iglu

def test_cgm2daybyday_basic():
    """Test basic functionality of cgm2daybyday"""
    
    # Create test data with known values
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',  # 0 min
            '2020-01-01 00:05:00',  # 5 min
            '2020-01-01 00:10:00',  # 10 min
            '2020-01-01 00:15:00'   # 15 min
        ]),
        'gl': [150, 200, 180, 160]
    })
    
    # Test with default parameters
    gd2d, dates, dt0 = iglu.cgm2daybyday(data)
    
    # Check output types and shapes
    assert isinstance(gd2d, np.ndarray)
    assert isinstance(dates, list)
    assert isinstance(dt0, int)
    assert gd2d.shape[0] == 1  # One day
    assert gd2d.shape[1] == 288  # 24 hours * 60 minutes / 5 minutes per measurement
    
    # Check that known values are preserved
    assert gd2d[0, 0] == 150  # First measurement
    assert gd2d[0, 1] == 200  # Second measurement
    assert gd2d[0, 2] == 180  # Third measurement
    assert gd2d[0, 3] == 160  # Fourth measurement
    
    # Check dates
    assert len(dates) == 1
    assert dates[0] == datetime(2020, 1, 1).date()
    
    # Check dt0
    assert dt0 == 5  # Default 5-minute intervals

def test_cgm2daybyday_multiple_days():
    """Test cgm2daybyday with multiple days of data"""
    
    # Create test data spanning multiple days
    data = pd.DataFrame({
        'id': ['subject1'] * 8,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 00:05:00',
            '2020-01-01 00:10:00', '2020-01-01 00:15:00',
            '2020-01-02 00:00:00', '2020-01-02 00:05:00',
            '2020-01-02 00:10:00', '2020-01-02 00:15:00'
        ]),
        'gl': [150, 200, 180, 160, 140, 190, 170, 210]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data)
    
    assert gd2d.shape[0] == 2  # Two days
    assert gd2d.shape[1] == 288  # 24 hours * 60 minutes / 5 minutes
    assert len(dates) == 2
    assert dates[0] == datetime(2020, 1, 1).date()
    assert dates[1] == datetime(2020, 1, 2).date()

def test_cgm2daybyday_custom_dt0():
    """Test cgm2daybyday with custom time interval"""
    
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:20:00',
            '2020-01-01 00:30:00'
        ]),
        'gl': [150, 200, 180, 160]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data, dt0=10)
    
    assert dt0 == 10
    assert gd2d.shape[1] == 144  # 24 hours * 60 minutes / 10 minutes

def test_cgm2daybyday_gaps():
    """Test cgm2daybyday with gaps in data"""
    
    data = pd.DataFrame({
        'id': ['subject1'] * 5,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',  # 0 min
            '2020-01-01 00:05:00',  # 5 min
            '2020-01-01 00:55:00',  # 55 min (gap > 45 min)
            '2020-01-01 01:00:00',  # 60 min
            '2020-01-01 01:05:00'   # 65 min
        ]),
        'gl': [150, 200, 180, 160, 170]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data, inter_gap=45)
    
    # Check that values in the gap are NaN
    gap_start_idx = 11  # 55 minutes / 5 minutes per measurement
    gap_end_idx = 12   # 60 minutes / 5 minutes per measurement
    assert np.all(np.isnan(gd2d[0, gap_start_idx:gap_end_idx]))

def test_cgm2daybyday_multiple_subjects():
    """Test cgm2daybyday with multiple subjects"""
    
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
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data)
    
    # Should only use first subject
    assert gd2d.shape[0] == 1
    assert gd2d[0, 0] == 150
    assert gd2d[0, 1] == 200

def test_cgm2daybyday_missing_values():
    """Test cgm2daybyday with missing values"""
    
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:15:00'
        ]),
        'gl': [150, np.nan, 180, 160]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data)
    
    # Check that missing values are handled appropriately
    assert not np.isnan(gd2d[0, 0])  # First value
    assert not np.isnan(gd2d[0, 2])  # Third value
    assert not np.isnan(gd2d[0, 3])  # Fourth value

def test_cgm2daybyday_unsorted_times():
    """Test cgm2daybyday with unsorted times"""
    
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:10:00',  # 10 min
            '2020-01-01 00:00:00',  # 0 min
            '2020-01-01 00:15:00',  # 15 min
            '2020-01-01 00:05:00'   # 5 min
        ]),
        'gl': [180, 150, 160, 200]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data)
    
    # Check that values are correctly ordered
    assert gd2d[0, 0] == 150  # First measurement
    assert gd2d[0, 1] == 200  # Second measurement
    assert gd2d[0, 2] == 180  # Third measurement
    assert gd2d[0, 3] == 160  # Fourth measurement

def test_cgm2daybyday_timezone():
    """Test cgm2daybyday with different timezone"""
    
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:15:00'
        ]),
        'gl': [150, 200, 180, 160]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data, tz='UTC')
    
    assert gd2d.shape[0] == 1
    assert gd2d.shape[1] == 288
    assert dates[0] == datetime(2020, 1, 1).date()

def test_cgm2daybyday_empty_data():
    """Test cgm2daybyday with empty data"""
    
    data = pd.DataFrame(columns=['id', 'time', 'gl'])
    
    with pytest.raises(IndexError):
        gd2d, dates, dt0 = iglu.cgm2daybyday(data)

def test_cgm2daybyday_single_measurement():
    """Test cgm2daybyday with only one measurement"""
    
    data = pd.DataFrame({
        'id': ['subject1'],
        'time': pd.to_datetime(['2020-01-01 00:00:00']),
        'gl': [150]
    })
    
    gd2d, dates, dt0 = iglu.cgm2daybyday(data)
    
    assert gd2d.shape[0] == 1
    assert gd2d.shape[1] == 288
    assert np.all(np.isnan(gd2d[0, 1:]))  # Only first value should be non-NaN
    assert gd2d[0, 0] == 150 