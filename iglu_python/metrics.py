import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

from .utils import check_data_columns, CGMS2DayByDay

def active_percent(data: pd.DataFrame, tz: str = "") -> pd.DataFrame:
    """
    Calculate the percentage of time CGM is active.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    tz : str, default=""
        Time zone to use for calculations
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id', 'active_percent', 'ndays', 'start_date', 'end_date'
    """
    data = check_data_columns(data)
    
    if tz:
        data['time'] = data['time'].dt.tz_localize(tz)
    
    result = []
    for subject in data['id'].unique():
        subject_data = data[data['id'] == subject]
        
        # Calculate total time span
        start_date = subject_data['time'].min()
        end_date = subject_data['time'].max()
        total_minutes = (end_date - start_date).total_seconds() / 60
        
        # Calculate active time (non-NA measurements)
        active_minutes = len(subject_data.dropna(subset=['gl'])) * 5  # Assuming 5-minute intervals
        
        # Calculate percentage
        active_percent = (active_minutes / total_minutes) * 100
        ndays = (end_date - start_date).days
        
        result.append({
            'id': subject,
            'active_percent': active_percent,
            'ndays': ndays,
            'start_date': start_date,
            'end_date': end_date
        })
    
    return pd.DataFrame(result)

def mean_glu(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean glucose values.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id' and 'mean'
    """
    data = check_data_columns(data)
    
    result = data.groupby('id')['gl'].mean().reset_index()
    result.columns = ['id', 'mean']
    return result

def gmi(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Glucose Management Indicator (GMI).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id' and 'GMI'
    """
    data = check_data_columns(data)
    
    # Calculate GMI using the formula: 3.31 + 0.02392 * mean glucose
    mean_glucose = mean_glu(data)
    mean_glucose['GMI'] = 3.31 + 0.02392 * mean_glucose['mean']
    
    return mean_glucose[['id', 'GMI']]

def cv_glu(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Coefficient of Variation (CV) of glucose values.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id' and 'CV'
    """
    data = check_data_columns(data)
    
    result = data.groupby('id').agg({
        'gl': lambda x: (x.std() / x.mean()) * 100
    }).reset_index()
    result.columns = ['id', 'CV']
    return result

def below_percent(data: pd.DataFrame, targets_below: List[float] = [54, 70]) -> pd.DataFrame:
    """
    Calculate percentage of time below target thresholds.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    targets_below : List[float], default=[54, 70]
        List of target thresholds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id' and percentage below each target
    """
    data = check_data_columns(data)
    
    result = []
    for subject in data['id'].unique():
        subject_data = data[data['id'] == subject]
        total_readings = len(subject_data.dropna(subset=['gl']))
        
        if total_readings == 0:
            continue
            
        percentages = {}
        for target in targets_below:
            below_count = len(subject_data[subject_data['gl'] < target])
            percentages[f'below_{target}'] = (below_count / total_readings) * 100
        
        percentages['id'] = subject
        result.append(percentages)
    
    return pd.DataFrame(result)

def in_range_percent(data: pd.DataFrame, target_ranges: List[List[float]] = [[70, 180]]) -> pd.DataFrame:
    """
    Calculate percentage of time in target ranges.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    target_ranges : List[List[float]], default=[[70, 180]]
        List of target ranges [lower, upper]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id' and percentage in each range
    """
    data = check_data_columns(data)
    
    result = []
    for subject in data['id'].unique():
        subject_data = data[data['id'] == subject]
        total_readings = len(subject_data.dropna(subset=['gl']))
        
        if total_readings == 0:
            continue
            
        percentages = {}
        for i, (lower, upper) in enumerate(target_ranges):
            in_range_count = len(subject_data[(subject_data['gl'] >= lower) & (subject_data['gl'] <= upper)])
            percentages[f'in_range_{lower}_{upper}'] = (in_range_count / total_readings) * 100
        
        percentages['id'] = subject
        result.append(percentages)
    
    return pd.DataFrame(result)

def above_percent(data: pd.DataFrame, targets_above: List[float] = [180, 250]) -> pd.DataFrame:
    """
    Calculate percentage of time above target thresholds.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    targets_above : List[float], default=[180, 250]
        List of target thresholds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id' and percentage above each target
    """
    data = check_data_columns(data)
    
    result = []
    for subject in data['id'].unique():
        subject_data = data[data['id'] == subject]
        total_readings = len(subject_data.dropna(subset=['gl']))
        
        if total_readings == 0:
            continue
            
        percentages = {}
        for target in targets_above:
            above_count = len(subject_data[subject_data['gl'] > target])
            percentages[f'above_{target}'] = (above_count / total_readings) * 100
        
        percentages['id'] = subject
        result.append(percentages)
    
    return pd.DataFrame(result)

def plot_ranges(data: pd.DataFrame) -> plt.Figure:
    """
    Create a bar plot showing time in ranges.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the ranges plot
    """
    data = check_data_columns(data)
    
    # Calculate percentages
    below_54 = below_percent(data, [54])['below_54'].iloc[0]
    below_70 = below_percent(data, [70])['below_70'].iloc[0]
    in_range = in_range_percent(data, [[70, 180]])['in_range_70_180'].iloc[0]
    above_180 = above_percent(data, [180])['above_180'].iloc[0]
    above_250 = above_percent(data, [250])['above_250'].iloc[0]
    
    # Calculate ranges
    ranges = {
        'Very Low (<54)': below_54,
        'Low (54-69)': below_70 - below_54,
        'Target (70-180)': in_range,
        'High (181-250)': above_180 - above_250,
        'Very High (>250)': above_250
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#8E1B1B', '#F92D00', '#48BA3C', '#F9F000', '#F9B500']
    
    # Create stacked bar
    bottom = 0
    for (label, value), color in zip(ranges.items(), colors):
        ax.bar(0, value, bottom=bottom, label=label, color=color)
        bottom += value
    
    # Customize plot
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel('Percentage')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks([])
    plt.tight_layout()
    
    return fig

def plot_agp(
    data: pd.DataFrame,
    LLTR: float = 70,
    ULTR: float = 180,
    smooth: bool = True,
    span: float = 0.3,
    dt0: Optional[datetime] = None,
    inter_gap: int = 45,
    tz: str = "",
    title: bool = False
) -> plt.Figure:
    """
    Create an Ambulatory Glucose Profile plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    LLTR : float, default=70
        Lower Limit of Target Range
    ULTR : float, default=180
        Upper Limit of Target Range
    smooth : bool, default=True
        Whether to smooth the quantiles
    span : float, default=0.3
        Span for loess smoothing
    dt0 : datetime, optional
        Start time for interpolation
    inter_gap : int, default=45
        Maximum gap (in minutes) between measurements to interpolate
    tz : str, default=""
        Time zone to use for calculations
    title : bool, default=False
        Whether to show the subject ID as title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the AGP plot
    """
    data = check_data_columns(data)
    
    # Interpolate data
    data_ip, actual_dates, dt0, _ = CGMS2DayByDay(data, dt0, inter_gap, tz)
    
    # Calculate quantiles
    quantiles = np.percentile(data_ip, [5, 25, 50, 75, 95], axis=0)
    
    # Create time points
    time_points = np.arange(0, 24, dt0/60)
    
    if smooth:
        from scipy.interpolate import interp1d
        # Smooth quantiles using loess-like approach
        smoothed_quantiles = []
        for q in quantiles:
            f = interp1d(time_points, q, kind='cubic')
            smoothed_quantiles.append(f(time_points))
        quantiles = np.array(smoothed_quantiles)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot quantiles
    ax.plot(time_points, quantiles[2], 'k-', label='Median', linewidth=2)
    ax.plot(time_points, quantiles[0], '--', color='#325DAA', label='5th percentile')
    ax.plot(time_points, quantiles[4], '--', color='#325DAA', label='95th percentile')
    
    # Fill areas
    ax.fill_between(time_points, quantiles[3], quantiles[4], color='#A7BEE7', alpha=0.5)
    ax.fill_between(time_points, quantiles[0], quantiles[1], color='#A7BEE7', alpha=0.5)
    ax.fill_between(time_points, quantiles[1], quantiles[3], color='#325DAA', alpha=0.5)
    
    # Add target range lines
    ax.axhline(y=LLTR, color='#48BA3C', linestyle='-')
    ax.axhline(y=ULTR, color='#48BA3C', linestyle='-')
    
    # Customize plot
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels(['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm', '12am'])
    ax.set_ylabel('Glucose [mg/dL]')
    ax.set_xlabel('Time of Day')
    
    if title:
        ax.set_title(data['id'].iloc[0])
    
    # Add quantile labels
    for i, q in enumerate([5, 25, 50, 75, 95]):
        ax.text(23.5, quantiles[i][-1], f'{q}%', ha='left', va='center')
    
    plt.tight_layout()
    return fig

def plot_daily(
    data: pd.DataFrame,
    maxd: int = 14,
    inter_gap: int = 45,
    tz: str = ""
) -> plt.Figure:
    """
    Create daily glucose profiles.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    maxd : int, default=14
        Maximum number of days to plot
    inter_gap : int, default=45
        Maximum gap (in minutes) between measurements to interpolate
    tz : str, default=""
        Time zone to use for calculations
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with daily profiles
    """
    data = check_data_columns(data)
    
    # Interpolate data
    data_ip, actual_dates, dt0, gaps = CGMS2DayByDay(data, None, inter_gap, tz)
    
    # Limit number of days
    n_days = min(maxd, len(actual_dates))
    data_ip = data_ip[:n_days]
    actual_dates = actual_dates[:n_days]
    
    # Create time points
    time_points = np.arange(0, 24, dt0/60)
    
    # Create plot
    fig, axes = plt.subplots(n_days, 1, figsize=(12, 2*n_days), sharex=True)
    if n_days == 1:
        axes = [axes]
    
    # Plot each day
    for i, (ax, date) in enumerate(zip(axes, actual_dates)):
        # Plot glucose values
        ax.plot(time_points, data_ip[i], 'b-', alpha=0.5)
        
        # Add target range
        ax.axhspan(70, 180, color='#48BA3C', alpha=0.2)
        
        # Customize subplot
        ax.set_ylim(50, 400)
        ax.set_ylabel(f'{date.date()}\nGlucose [mg/dL]')
        if i == n_days - 1:
            ax.set_xlabel('Time of Day')
            ax.set_xticks(range(0, 25, 3))
            ax.set_xticklabels(['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm', '12am'])
    
    plt.tight_layout()
    return fig 