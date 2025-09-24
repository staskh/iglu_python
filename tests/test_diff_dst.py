"""
Test suite for testing diff() functionality with DST (Daylight Saving Time) transitions.

This module tests how data.index.to_series().diff() behaves with timezone-aware data
and demonstrates the correct way to create timezone-aware timestamps.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import pytz


class TestDiffDST:
    """Test class for diff() functionality with DST transitions."""

    def test_correct_timezone_creation(self):
        """Test that using tz.localize() gives correct round offsets."""
        tz = pytz.timezone('US/Eastern')
        
        # Correct approach: use tz.localize()
        correct_timestamps = [
            tz.localize(datetime(2024, 3, 10, 1, 0)),   # 1:00 AM EST
            tz.localize(datetime(2024, 6, 15, 10, 0)),  # 10:00 AM EDT
            tz.localize(datetime(2024, 12, 15, 10, 0)), # 10:00 AM EST
        ]
        
        # Verify round offsets
        offsets = [ts.utcoffset().total_seconds() / 3600 for ts in correct_timestamps]
        assert offsets[0] == -5.0  # EST
        assert offsets[1] == -4.0  # EDT  
        assert offsets[2] == -5.0  # EST
        
        # Wrong approach: direct tzinfo assignment (would give non-round offsets)
        # datetime(2024, 3, 10, 1, 0, tzinfo=tz)  # This gives -4.933333 hours

    def test_regular_intervals_no_dst(self):
        """Test diff() behavior with regular intervals during non-DST periods."""
        # Regular 15-minute intervals during summer (EDT)
        tz = pytz.timezone('US/Eastern')
        timestamps = [
            tz.localize(datetime(2024, 6, 15, 10, 0)),   # 10:00 AM EDT
            tz.localize(datetime(2024, 6, 15, 10, 15)),  # 10:15 AM EDT
            tz.localize(datetime(2024, 6, 15, 10, 30)),  # 10:30 AM EDT
            tz.localize(datetime(2024, 6, 15, 10, 45)),  # 10:45 AM EDT
        ]
        
        # Create DataFrame with timezone-aware index
        data = pd.DataFrame({
            'gl': [100, 110, 120, 130]
        }, index=timestamps)
        
        # Calculate time differences using diff()
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
        
        # Verify continuous monotonic time increase
        valid_diffs = time_diffs.dropna()
        assert all(valid_diffs > 0), f"Found negative time differences: {valid_diffs[valid_diffs <= 0]}"
        
        # Verify expected differences (all 15 minutes)
        expected_diffs = [np.nan, 15.0, 15.0, 15.0]
        np.testing.assert_allclose(time_diffs, expected_diffs, rtol=1e-10)
        
        # Verify median calculation
        median_diff = np.nanmedian(time_diffs)
        assert median_diff == 15.0
        
        # Verify that timestamps are monotonically increasing
        assert data.index.is_monotonic_increasing
        
        # Verify consistent UTC offset (EDT = UTC-4)
        utc_offsets = [ts.utcoffset().total_seconds() / 3600 for ts in timestamps]
        assert all(offset == -4.0 for offset in utc_offsets)  # All should be EDT

    def test_naive_timestamps_dst_problem(self):
        """Test that naive timestamps can cause issues during DST transitions."""
        # Create naive timestamps during DST transition (problematic case)
        naive_timestamps = [
            datetime(2024, 3, 10, 1, 30),  # 1:30 AM (naive)
            datetime(2024, 3, 10, 2, 0),   # 2:00 AM (naive) - ambiguous!
            datetime(2024, 3, 10, 3, 0),   # 3:00 AM (naive) - ambiguous!
            datetime(2024, 3, 10, 3, 30),  # 3:30 AM (naive)
        ]
        
        # Create DataFrame with naive index
        data = pd.DataFrame({
            'gl': [100, 110, 120, 130]
        }, index=naive_timestamps)
        
        # Verify no timezone information
        assert data.index.tz is None
        
        # Calculate time differences using diff()
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
        
        # Verify continuous monotonic time increase
        valid_diffs = time_diffs.dropna()
        assert all(valid_diffs > 0), f"Found negative time differences: {valid_diffs[valid_diffs <= 0]}"
        
        # Verify expected differences (naive timestamps don't account for DST)
        expected_diffs = [np.nan, 30.0, 60.0, 30.0]  # minutes
        np.testing.assert_allclose(time_diffs, expected_diffs, rtol=1e-10)
        
        # Verify median calculation
        median_diff = np.nanmedian(time_diffs)
        assert median_diff == 30.0
        
        # Verify that timestamps are monotonically increasing
        assert data.index.is_monotonic_increasing

    def test_dst_diff_behavior_works_correctly(self):
        """Test that diff() works correctly with timezone-aware data."""
        # This test demonstrates that diff() works correctly with timezone-aware data
        tz = pytz.timezone('US/Eastern')
        
        # Create a realistic scenario: measurements every 15 minutes
        timestamps = [
            tz.localize(datetime(2024, 3, 10, 1, 0)),   # 1:00 AM EST
            tz.localize(datetime(2024, 3, 10, 1, 15)),  # 1:15 AM EST
            tz.localize(datetime(2024, 3, 10, 1, 30)),  # 1:30 AM EST
            tz.localize(datetime(2024, 3, 10, 1, 45)),  # 1:45 AM EST
            tz.localize(datetime(2024, 3, 10, 3, 0)),   # 3:00 AM EDT
            tz.localize(datetime(2024, 3, 10, 3, 15)),  # 3:15 AM EDT
            tz.localize(datetime(2024, 3, 10, 3, 30)),  # 3:30 AM EDT
        ]
        
        data = pd.DataFrame({
            'gl': [100, 110, 120, 130, 140, 150, 160]
        }, index=timestamps)
        
        # Calculate time differences using diff()
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
        
        # Verify that timestamps are monotonically increasing
        assert data.index.is_monotonic_increasing
        
        # Verify that all differences are positive (time always moves forward)
        valid_diffs = time_diffs.dropna()
        assert all(valid_diffs > 0), f"Found negative time differences: {valid_diffs[valid_diffs <= 0]}"
        
        # Verify median calculation works
        median_diff = np.nanmedian(time_diffs)
        assert median_diff == 15.0  # Most intervals are 15 minutes

    def test_timezone_offset_demonstration(self):
        """Demonstrate the difference between correct and incorrect timezone creation."""
        tz = pytz.timezone('US/Eastern')
        
        # Correct approach
        correct_ts = tz.localize(datetime(2024, 3, 10, 1, 0))
        correct_offset = correct_ts.utcoffset().total_seconds() / 3600
        
        # Incorrect approach (would give non-round offset)
        incorrect_ts = datetime(2024, 3, 10, 1, 0, tzinfo=tz)
        incorrect_offset = incorrect_ts.utcoffset().total_seconds() / 3600
        
        # Demonstrate the difference
        assert correct_offset == -5.0  # Round number
        assert abs(incorrect_offset - (-4.933333)) < 0.001  # Non-round number
        
        print(f"Correct approach: {correct_offset} hours")
        print(f"Incorrect approach: {incorrect_offset} hours")

    def test_est_to_edt_transition_15min_intervals(self):
        """Test diff() behavior crossing EST->EDT transition with 15-minute intervals."""
        # Spring Forward: March 10, 2024 at 2:00 AM EST -> 3:00 AM EDT
        tz = pytz.timezone('US/Eastern')
        
        # Create timestamps that cross the DST transition
        timestamps = [
            tz.localize(datetime(2024, 3, 10, 1, 45)),  # 1:45 AM EST
            tz.localize(datetime(2024, 3, 10, 2, 0)),   # 2:00 AM EST -> 3:00 AM EDT
            tz.localize(datetime(2024, 3, 10, 3, 0)),   # 3:00 AM EDT (spring forward)
            tz.localize(datetime(2024, 3, 10, 3, 15)),  # 3:15 AM EDT
            tz.localize(datetime(2024, 3, 10, 3, 30)),  # 3:30 AM EDT
        ]
        
        # Create DataFrame with timezone-aware index
        data = pd.DataFrame({
            'gl': [100 + i*10 for i in range(5)]
        }, index=timestamps)
        
        # Calculate time differences using diff()
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
        
        # Verify that timestamps are monotonically increasing
        assert data.index.is_monotonic_increasing
        
        # Verify that all differences are non-negative (time always moves forward)
        valid_diffs = time_diffs.dropna()
        assert all(valid_diffs >= 0), f"Found negative time differences: {valid_diffs[valid_diffs < 0]}"
        
        # Verify expected differences (all should be 15 minutes)
        expected_diffs = [np.nan, 15.0, 0.0, 15.0, 15.0]  # First is NaN, rest are 15 minutes except DST gap
        np.testing.assert_allclose(time_diffs, expected_diffs, rtol=1e-10)
        
        # Verify median calculation
        median_diff = np.nanmedian(time_diffs)
        assert median_diff == 15.0
        
        # Verify UTC offset changes during DST transition
        utc_offsets = [ts.utcoffset().total_seconds() / 3600 for ts in timestamps]
        # First 2 timestamps should be EST (-5.0), last 3 should be EDT (-4.0)
        assert all(offset == -5.0 for offset in utc_offsets[:2])  # EST
        assert all(offset == -4.0 for offset in utc_offsets[2:])   # EDT
        
        print(f"EST->EDT transition timestamps:")
        for i, ts in enumerate(timestamps):
            print(f"  {i+1}: {ts} (UTC offset: {ts.utcoffset().total_seconds()/3600:.1f})")
        print(f"Time differences: {time_diffs.tolist()}")

    def test_edt_to_est_transition_15min_intervals(self):
        """Test diff() behavior crossing EDT->EST transition with 15-minute intervals."""
        # Fall Back: November 3, 2024 at 2:00 AM EDT -> 1:00 AM EST
        tz = pytz.timezone('US/Eastern')
        
        # Create timestamps that cross the DST transition
        timestamps = [
            tz.localize(datetime(2024, 11, 3, 1, 45)),  # 1:45 AM EDT
            tz.localize(datetime(2024, 11, 3, 2, 0)),   # 2:00 AM EDT -> 1:00 AM EST
            tz.localize(datetime(2024, 11, 3, 1, 0)),   # 1:00 AM EST (fall back)
            tz.localize(datetime(2024, 11, 3, 1, 15)),  # 1:15 AM EST
            tz.localize(datetime(2024, 11, 3, 1, 30)),  # 1:30 AM EST
        ]
        
        # Sort timestamps to ensure chronological order
        timestamps.sort()
        
        # Create DataFrame with timezone-aware index
        data = pd.DataFrame({
            'gl': [100 + i*10 for i in range(5)]
        }, index=timestamps)
        
        # Calculate time differences using diff()
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
        
        # Verify that timestamps are monotonically increasing
        assert data.index.is_monotonic_increasing
        
        # Verify that all differences are non-negative (time always moves forward)
        valid_diffs = time_diffs.dropna()
        assert all(valid_diffs >= 0), f"Found negative time differences: {valid_diffs[valid_diffs < 0]}"
        
        # Verify expected differences (all should be 15 minutes)
        expected_diffs = [np.nan, 15.0, 15.0, 15.0, 15.0]  # First is NaN, rest are 15 minutes
        np.testing.assert_allclose(time_diffs, expected_diffs, rtol=1e-10)
        
        # Verify median calculation
        median_diff = np.nanmedian(time_diffs)
        assert median_diff == 15.0
        
        # Verify UTC offset changes during DST transition
        utc_offsets = [ts.utcoffset().total_seconds() / 3600 for ts in timestamps]
        # All timestamps should be EST (-5.0) after sorting
        assert all(offset == -5.0 for offset in utc_offsets)  # EST
        
        print(f"EDT->EST transition timestamps:")
        for i, ts in enumerate(timestamps):
            print(f"  {i+1}: {ts} (UTC offset: {ts.utcoffset().total_seconds()/3600:.1f})")
        print(f"Time differences: {time_diffs.tolist()}")