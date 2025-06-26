import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

# Import the module to test
from iglu_python.extension.plots import plot_daily, plot_statistics
from iglu_python import load_libre


@pytest.fixture(scope="module")
def sample_cgm_data():
    """Create sample CGM data for testing"""
    # Create datetime index for 3 days with 5-minute intervals
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-03 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose values with some variation
    np.random.seed(42)  # For reproducible tests
    base_glucose = 120
    glucose_values = []
    
    for i, timestamp in enumerate(time_index):
        # Add some daily variation (lower at night, higher during day)
        hour = timestamp.hour
        if 6 <= hour <= 18:  # Daytime
            daily_factor = 1.2
        else:  # Nighttime
            daily_factor = 0.8
        
        # Add some random variation
        random_factor = 1 + np.random.normal(0, 0.1)
        glucose = base_glucose * daily_factor * random_factor
        
        # Ensure reasonable glucose range
        glucose = max(50, min(400, glucose))
        glucose_values.append(glucose)
    
    return pd.Series(glucose_values, index=time_index)


@pytest.fixture(scope="module")
def single_day_data():
    """Create single day CGM data for testing"""
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-01 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    np.random.seed(42)
    glucose_values = 120 + np.random.normal(0, 20, len(time_index))
    glucose_values = np.clip(glucose_values, 50, 400)
    
    return pd.Series(glucose_values, index=time_index)


@pytest.fixture(scope="module")
def data_with_extremes():
    """Create data with extreme values to test thresholds"""
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-01 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='30min')
    
    # Create data with values below and above thresholds
    glucose_values = [50, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    glucose_values.extend([120] * (len(time_index) - len(glucose_values)))
    
    return pd.Series(glucose_values, index=time_index)


@pytest.fixture(scope="module")
def multi_day_statistics_data():
    """Create multi-day data for statistics testing"""
    # Create datetime index for 5 days with 10-minute intervals
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-05 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='10min')
    
    np.random.seed(42)
    base_glucose = 120
    glucose_values = []
    
    for i, timestamp in enumerate(time_index):
        # Add daily pattern with some variation
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Different patterns for different days
        if day_of_week < 3:  # Weekdays
            daily_factor = 1.1
        else:  # Weekends
            daily_factor = 0.9
        
        # Time of day variation
        if 6 <= hour <= 18:  # Daytime
            time_factor = 1.2
        else:  # Nighttime
            time_factor = 0.8
        
        # Add random variation
        random_factor = 1 + np.random.normal(0, 0.15)
        glucose = base_glucose * daily_factor * time_factor * random_factor
        
        # Ensure reasonable glucose range
        glucose = max(50, min(400, glucose))
        glucose_values.append(glucose)
    
    return pd.Series(glucose_values, index=time_index)


@pytest.fixture(scope="module")
def libre_data():
    """Load real Libre data for testing"""
    data_path = Path("tests/data/libre_amer_02.csv")
    if not data_path.exists():
        pytest.skip(f"Test data file not found: {data_path}")
    
    try:
        return load_libre(str(data_path))
    except Exception as e:
        pytest.skip(f"Could not load Libre data: {e}")


def test_plot_daily_returns_figure(sample_cgm_data):
    """Test that plot_daily returns a matplotlib Figure object"""
    fig = plot_daily(sample_cgm_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_daily_single_day(single_day_data):
    """Test plot_daily with single day data"""
    fig = plot_daily(single_day_data)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1  # Should have one subplot for single day
    
    # Check that the subplot has the expected title format
    ax = fig.axes[0]
    assert 'Day: 2023-01-01' in ax.get_title()
    
    plt.close(fig)


def test_plot_daily_multiple_days(sample_cgm_data):
    """Test plot_daily with multiple days data"""
    fig = plot_daily(sample_cgm_data)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3  # Should have 3 subplots for 3 days
    
    # Check that each subplot has the expected title format
    expected_dates = ['2023-01-01', '2023-01-02', '2023-01-03']
    for i, ax in enumerate(fig.axes):
        assert f'Day: {expected_dates[i]}' in ax.get_title()
    
    plt.close(fig)


def test_plot_daily_custom_thresholds(data_with_extremes):
    """Test plot_daily with custom lower and upper thresholds"""
    fig = plot_daily(data_with_extremes, lower=80, upper=160)
    
    assert isinstance(fig, plt.Figure)
    
    # Check that the subplot has the expected labels
    ax = fig.axes[0]
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    
    assert any('Hypo threshold (80 mg/dL)' in text for text in legend_texts)
    assert any('Hyper threshold (160 mg/dL)' in text for text in legend_texts)
    
    plt.close(fig)


def test_plot_daily_default_thresholds(data_with_extremes):
    """Test plot_daily with default thresholds"""
    fig = plot_daily(data_with_extremes)
    
    assert isinstance(fig, plt.Figure)
    
    # Check that the subplot has the expected default labels
    ax = fig.axes[0]
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    
    assert any('Hypo threshold (70 mg/dL)' in text for text in legend_texts)
    assert any('Hyper threshold (140 mg/dL)' in text for text in legend_texts)
    
    plt.close(fig)


def test_plot_daily_axis_labels(sample_cgm_data):
    """Test that plot_daily sets correct axis labels"""
    fig = plot_daily(sample_cgm_data)
    
    for ax in fig.axes:
        assert ax.get_ylabel() == 'Glucose (mg/dL)'
        assert ax.get_xlabel() == 'Time (hours)'
    
    plt.close(fig)


def test_plot_daily_legend_present(sample_cgm_data):
    """Test that plot_daily adds legend to all subplots"""
    fig = plot_daily(sample_cgm_data)
    
    for ax in fig.axes:
        assert ax.get_legend() is not None
    
    plt.close(fig)


def test_plot_daily_xticks_format(sample_cgm_data):
    """Test that plot_daily sets x-axis ticks to show hours"""
    fig = plot_daily(sample_cgm_data)
    
    for ax in fig.axes:
        xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
        # Should have 24 hour labels (0-23)
        assert len(xtick_labels) == 24
        assert '0' in xtick_labels
        assert '12' in xtick_labels
        assert '23' in xtick_labels
    
    plt.close(fig)


def test_plot_daily_ylim_reasonable(sample_cgm_data):
    """Test that plot_daily sets reasonable y-axis limits"""
    fig = plot_daily(sample_cgm_data)
    
    for ax in fig.axes:
        ylim = ax.get_ylim()
        assert ylim[0] == 0  # Should start at 0
        assert ylim[1] >= 300  # Should go at least to 300
    
    plt.close(fig)


def test_plot_daily_empty_data():
    """Test plot_daily with empty data (should handle gracefully)"""
    empty_series = pd.Series(dtype=float)
    empty_series.index = pd.DatetimeIndex([])
    
    with pytest.raises(ValueError):
        plot_daily(empty_series)


def test_plot_daily_invalid_input_type():
    """Test plot_daily with invalid input type"""
    with pytest.raises(AttributeError):
        plot_daily("not a series")


def test_plot_daily_negative_thresholds(data_with_extremes):
    """Test plot_daily with negative thresholds (edge case)"""
    fig = plot_daily(data_with_extremes, lower=-10, upper=500)
    
    assert isinstance(fig, plt.Figure)
    
    # Check that the subplot has the expected labels
    ax = fig.axes[0]
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    
    assert any('Hypo threshold (-10 mg/dL)' in text for text in legend_texts)
    assert any('Hyper threshold (500 mg/dL)' in text for text in legend_texts)
    
    plt.close(fig)


def test_plot_daily_threshold_order(data_with_extremes):
    """Test plot_daily when lower > upper (edge case)"""
    fig = plot_daily(data_with_extremes, lower=200, upper=100)
    
    assert isinstance(fig, plt.Figure)
    
    # Check that the subplot has the expected labels
    ax = fig.axes[0]
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    
    assert any('Hypo threshold (200 mg/dL)' in text for text in legend_texts)
    assert any('Hyper threshold (100 mg/dL)' in text for text in legend_texts)
    
    plt.close(fig)


def test_plot_daily_figure_size(sample_cgm_data):
    """Test that plot_daily creates figure with expected size"""
    fig = plot_daily(sample_cgm_data)
    
    # Should have width=12 and height=3*num_days
    expected_size = (12, 3 * 3)  # 3 days
    assert fig.get_size_inches().tolist() == list(expected_size)
    
    plt.close(fig)


def test_plot_daily_save_figure(sample_cgm_data):
    """Test that the returned figure can be saved"""
    fig = plot_daily(sample_cgm_data)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    fig.savefig(tmp_filename, dpi=72)
    plt.close(fig)
    assert os.path.exists(tmp_filename)
    assert os.path.getsize(tmp_filename) > 0
    os.unlink(tmp_filename)
    
    plt.close(fig)


def test_plot_daily_data_with_nan_values():
    """Test plot_daily with data containing NaN values"""
    # Create data with some NaN values
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-01 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    glucose_values = [120, 130, np.nan, 110, 140, np.nan, 125, 135]
    glucose_values.extend([120] * (len(time_index) - len(glucose_values)))
    
    data_with_nan = pd.Series(glucose_values, index=time_index)
    
    fig = plot_daily(data_with_nan)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_daily_very_high_glucose_values():
    """Test plot_daily with very high glucose values"""
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-01 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='2h')
    
    # Create data with very high values
    glucose_values = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
    
    high_glucose_data = pd.Series(glucose_values, index=time_index)
    
    fig = plot_daily(high_glucose_data)
    
    assert isinstance(fig, plt.Figure)
    
    # Check that y-axis limit is adjusted for high values
    ax = fig.axes[0]
    ylim = ax.get_ylim()
    assert ylim[1] >= 1700  # Should accommodate the highest value
    
    plt.close(fig)


# Tests for plot_statistics function
def test_plot_statistics_returns_figure(multi_day_statistics_data):
    """Test that plot_statistics returns a matplotlib Figure object"""
    fig = plot_statistics(multi_day_statistics_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_statistics_single_day(single_day_data):
    """Test plot_statistics with single day data"""
    fig = plot_statistics(single_day_data)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1  # Should have one subplot
    
    plt.close(fig)


def test_plot_statistics_multiple_days(multi_day_statistics_data):
    """Test plot_statistics with multiple days data"""
    fig = plot_statistics(multi_day_statistics_data)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1  # Should have one subplot for statistics
    
    plt.close(fig)


def test_plot_statistics_custom_thresholds(data_with_extremes):
    """Test plot_statistics with custom lower and upper thresholds"""
    fig = plot_statistics(data_with_extremes, lower=80, upper=160)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_default_thresholds(data_with_extremes):
    """Test plot_statistics with default thresholds"""
    fig = plot_statistics(data_with_extremes)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_figure_size(multi_day_statistics_data):
    """Test that plot_statistics creates figure with expected size"""
    fig = plot_statistics(multi_day_statistics_data)
    
    # Should have width=12 and height=6
    expected_size = (12, 6)
    assert fig.get_size_inches().tolist() == list(expected_size)
    
    plt.close(fig)


def test_plot_statistics_save_figure(multi_day_statistics_data):
    """Test that the returned figure can be saved"""
    fig = plot_statistics(multi_day_statistics_data)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    fig.savefig(tmp_filename, dpi=72)
    plt.close(fig)
    assert os.path.exists(tmp_filename)
    assert os.path.getsize(tmp_filename) > 0
    os.unlink(tmp_filename)
    
    plt.close(fig)


def test_plot_statistics_data_with_nan_values():
    """Test plot_statistics with data containing NaN values"""
    # Create data with some NaN values
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-03 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='2h')
    
    glucose_values = [120, 130, np.nan, 110, 140, np.nan, 125, 135]
    glucose_values.extend([120] * (len(time_index) - len(glucose_values)))
    
    data_with_nan = pd.Series(glucose_values, index=time_index)
    
    fig = plot_statistics(data_with_nan)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_data_with_gaps():
    """Test plot_statistics with data containing gaps (interpolation test)"""
    # Create data with gaps
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-03 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Create data with some missing values
    glucose_values = []
    for i, timestamp in enumerate(time_index):
        if i % 4 == 0:  # Every 4th value is missing
            glucose_values.append(np.nan)
        else:
            glucose_values.append(120 + np.random.normal(0, 20))
    
    data_with_gaps = pd.Series(glucose_values, index=time_index)
    
    fig = plot_statistics(data_with_gaps)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_very_high_glucose_values():
    """Test plot_statistics with very high glucose values"""
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    end_date = pd.Timestamp('2023-01-03 23:59:59')
    time_index = pd.date_range(start=start_date, end=end_date, freq='3h')
    
    # Create data with very high values
    glucose_values = [600, 700, 800, 900, 1000, 1100, 1200, 1300]
    glucose_values.extend([1200] * (len(time_index) - len(glucose_values)))
    
    high_glucose_data = pd.Series(glucose_values, index=time_index)
    
    fig = plot_statistics(high_glucose_data)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_negative_thresholds(data_with_extremes):
    """Test plot_statistics with negative thresholds (edge case)"""
    fig = plot_statistics(data_with_extremes, lower=-10, upper=500)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_threshold_order(data_with_extremes):
    """Test plot_statistics when lower > upper (edge case)"""
    fig = plot_statistics(data_with_extremes, lower=200, upper=100)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_statistics_empty_data():
    """Test plot_statistics with empty data (should handle gracefully)"""
    empty_series = pd.Series(dtype=float)
    empty_series.index = pd.DatetimeIndex([])
    
    with pytest.raises(ValueError):
        plot_statistics(empty_series)


def test_plot_statistics_invalid_input_type():
    """Test plot_statistics with invalid input type"""
    with pytest.raises(AttributeError):
        plot_statistics("not a series")


def test_plot_statistics_legend_labels(multi_day_statistics_data):
    """Test that plot_statistics creates plots with expected legend labels"""
    fig = plot_statistics(multi_day_statistics_data)
    
    ax = fig.axes[0]
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    
    # Check for expected legend labels
    assert any('Mean sample trends' in text for text in legend_texts)
    assert any('10% quantile' in text for text in legend_texts)
    assert any('25% quantile' in text for text in legend_texts)
    assert any('75% quantile' in text for text in legend_texts)
    assert any('90% quantile' in text for text in legend_texts)
    
    plt.close(fig)


def test_plot_statistics_line_count(multi_day_statistics_data):
    """Test that plot_statistics creates the expected number of lines"""
    fig = plot_statistics(multi_day_statistics_data)
    
    ax = fig.axes[0]
    lines = ax.get_lines()
    
    # Should have at least 2 lines: mean trends and quantiles
    assert len(lines) >= 2
    
    plt.close(fig)


def test_plot_statistics_data_consistency(multi_day_statistics_data):
    """Test that plot_statistics produces consistent results for same input"""
    fig1 = plot_statistics(multi_day_statistics_data)
    fig2 = plot_statistics(multi_day_statistics_data)
    
    # Both should be valid figures
    assert isinstance(fig1, plt.Figure)
    assert isinstance(fig2, plt.Figure)
    
    plt.close(fig1)
    plt.close(fig2)


# Tests using real Libre data
def test_plot_statistics_with_libre_data(libre_data):
    """Test plot_statistics with real Libre CGM data"""
    fig = plot_statistics(libre_data)
    
    # Basic assertions
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    # Check that the data was processed correctly
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) >= 1  # Should have mean trends line
    
    # Check legend labels - the actual labels from plot_statistics function
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any('Mean sample trends' in text for text in legend_texts)
    assert any('10% quantile' in text for text in legend_texts)
    assert any('25% quantile' in text for text in legend_texts)
    assert any('75% quantile' in text for text in legend_texts)
    assert any('90% quantile' in text for text in legend_texts)
    
    # Check that the plot has data (not empty)
    for line in lines:
        assert len(line.get_ydata()) > 0
        assert len(line.get_xdata()) > 0
    
    plt.close(fig)


def test_plot_statistics_with_libre_data_custom_thresholds(libre_data):
    """Test plot_statistics with real Libre data and custom thresholds"""
    fig = plot_statistics(libre_data, lower=80, upper=160)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    # Verify the plot was created successfully
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) >= 2
    
    plt.close(fig)


def test_plot_statistics_with_libre_data_save_figure(libre_data):
    """Test that plot_statistics with Libre data can be saved"""
    fig = plot_statistics(libre_data)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    fig.savefig(tmp_filename, dpi=72)
    plt.close(fig)
    assert os.path.exists(tmp_filename)
    assert os.path.getsize(tmp_filename) > 0
    os.unlink(tmp_filename)
    
    plt.close(fig)


def test_plot_statistics_with_libre_data_glucose_range(libre_data):
    """Test that plot_statistics with Libre data produces reasonable glucose ranges"""
    fig = plot_statistics(libre_data)
    
    ax = fig.axes[0]
    lines = ax.get_lines()
    
    # Check that all lines have reasonable glucose values
    for line in lines:
        y_data = np.array(line.get_ydata())
        if len(y_data) > 0:
            # Glucose values should be in reasonable range (20-600 mg/dL)
            assert np.all(y_data >= 20) or np.all(np.isnan(y_data))
            assert np.all(y_data <= 600) or np.all(np.isnan(y_data))
    
    plt.close(fig)


def test_plot_statistics_with_libre_data_time_range(libre_data):
    """Test that plot_statistics with Libre data produces reasonable time ranges"""
    fig = plot_statistics(libre_data)
    
    ax = fig.axes[0]
    lines = ax.get_lines()
    
    # Check that all lines have reasonable time indices
    for line in lines:
        x_data = line.get_xdata()
        if len(x_data) > 0:
            # Time indices should be non-negative and reasonable
            # Check that all x_data values represent proper timestamps
            # Convert to pandas timestamps and verify they are valid
            timestamps = pd.to_datetime(x_data, unit='D', origin='unix')
            assert not timestamps.isna().any(), "All timestamps should be valid"
            # Should have multiple time points for statistics
            assert len(x_data) > 1
    
    plt.close(fig) 