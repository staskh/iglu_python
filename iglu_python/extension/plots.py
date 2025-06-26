"""
This module implements various plots for the iglu_python package.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_daily(cgm_timeseries: pd.Series, lower: int = 70, upper: int = 140) -> plt.Figure:
    """
    Plot daily Glucose values for each day separately



    Args:
        - cgm_timeseries: pd.Series
        - lower: int, default=70, Lower bound used for hypoglycemia cutoff, in mg/dL
        - upper: int, default=140, Upper bound used for hyperglycemia cutoff, in mg/dL

    Returns:
        plt.Figure object
    """
    # divide cgm_timeseries into list of daily series
    cgm_daily_group = cgm_timeseries.resample("D")
    cgm_timeseries_daily = {day: cgm_daily_group.get_group(day) for day in cgm_daily_group.groups}

    # plot each day separately
    # Create one figure with subplots for each day
    num_days = len(cgm_timeseries_daily)
    fig, axes = plt.subplots(num_days, 1, figsize=(12, 3 * num_days))

    # If only one day, axes will be a single object, not an array
    if num_days == 1:
        axes = [axes]

    for i, (day, cgm_one_day) in enumerate(cgm_timeseries_daily.items()):
        # Convert datetime index to time-only for x-axis display
        axes[i].plot(cgm_one_day.index, cgm_one_day.values)
        axes[i].set_title(f"Day: {day.strftime('%Y-%m-%d')}")
        axes[i].set_ylabel("Glucose (mg/dL)")
        axes[i].set_ylim(0, max(np.nanmax(cgm_one_day.values), 300))

        # Fill area above upper limit and plot it in orange
        upper_array = [upper] * len(cgm_one_day.values)
        area_over_upper = [
            cgm_one_day.values[i] if cgm_one_day.values[i] > upper else upper for i in range(len(cgm_one_day.values))
        ]
        axes[i].fill_between(cgm_one_day.index, area_over_upper, upper_array, alpha=0.3, color="orange")
        axes[i].axhline(y=upper, color="orange", linestyle="--", alpha=0.7, label=f"Hyper threshold ({upper} mg/dL)")

        # Fill area below lower  limit and plot it in blue
        lower_array = [lower] * len(cgm_one_day.values)
        area_below_lower = [
            cgm_one_day.values[i] if cgm_one_day.values[i] < lower else lower for i in range(len(cgm_one_day.values))
        ]
        axes[i].fill_between(cgm_one_day.index, lower_array, area_below_lower, alpha=0.3, color="blue")
        axes[i].axhline(y=lower, color="blue", linestyle="--", alpha=0.7, label=f"Hypo threshold ({lower} mg/dL)")

        # on horisontal axis, show only time in hours
        axes[i].set_xlabel("Time (hours)")
        time_range = pd.date_range(start=day, periods=24, freq="1h")
        axes[i].set_xticks(time_range)  # Show every hour from 0 to 24
        axes[i].set_xticklabels([f"{h.hour}" for h in time_range])  # Format as HH:00
        axes[i].grid(True, alpha=0.3, linestyle="--")
        axes[i].legend()

    fig.tight_layout()
    return fig


def plot_statistics(cgm_timeseries: pd.Series, lower: int = 70, upper: int = 140) -> plt.Figure:
    """
    Plot statistical representation of daily trends
    in the single 24h timeline, this will plot mean sample trends, 10%, +25% and 75% and 90% quantiles
    """
    # check if cgm_timeseries is a pandas series
    if not isinstance(cgm_timeseries, pd.Series):
        raise AttributeError("cgm_timeseries must be a pandas series")

    # check if cgm_timeseries is not a datetime index
    if not isinstance(cgm_timeseries.index, pd.DatetimeIndex):
        raise AttributeError("cgm_timeseries must have a datetime index")

    # check if cgm_timeseries is not empty
    if len(cgm_timeseries) < 16:
        raise ValueError("cgm_timeseries is too short to plot statistics")

    # get sampling frequency
    time_diffs = cgm_timeseries.index.to_series().diff()
    dt0 = int(time_diffs.mode().iloc[0].total_seconds() / 60)

    # Create time grid
    start_time = cgm_timeseries.index.min().floor("D")
    end_time = cgm_timeseries.index.max().ceil("D")
    time_grid = pd.date_range(start=start_time, end=end_time, freq=f"{dt0}min")
    # remove the last time point
    time_grid = time_grid[:-1]

    # interpolate
    cgm_timeseries_interpolated = np.interp(
        (time_grid - start_time).total_seconds() / 60,
        (cgm_timeseries.index - start_time).total_seconds() / 60,
        cgm_timeseries.values,
        left=np.nan,
        right=np.nan,
    )

    # reorganise as 2d array with rows as timepoints and columns as days
    # Reshape to days
    n_days = (end_time - start_time).days
    n_points_per_day = 24 * 60 // dt0
    cgm_timeseries_2d = cgm_timeseries_interpolated.reshape(n_days, n_points_per_day)

    # one day time grid
    time_grid_one_day = time_grid[0:n_points_per_day]
    # get mean sample trends
    mean_sample_trends = np.nanmean(cgm_timeseries_2d, axis=0)

    # get 10%, +25% and 75% and 90% quantiles
    quantiles = np.nanpercentile(cgm_timeseries_2d, [10, 25, 75, 90], axis=0)

    # create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # plot mean sample trends
    ax.plot(time_grid_one_day, mean_sample_trends, color="orange", alpha=1, linewidth=3, label="Mean sample trends")

    # plot quantiles
    ax.fill_between(time_grid_one_day, quantiles[0], quantiles[1], alpha=0.25, color="blue", label="10% quantile")
    ax.fill_between(time_grid_one_day, quantiles[1], mean_sample_trends, alpha=0.50, color="blue", label="25% quantile")
    ax.fill_between(time_grid_one_day, mean_sample_trends, quantiles[2], alpha=0.50, color="blue", label="75% quantile")
    ax.fill_between(time_grid_one_day, quantiles[2], quantiles[3], alpha=0.25, color="blue", label="90% quantile")

    ax.axhline(y=upper, color="orange", linestyle="--", alpha=0.7, label=f"Hyper threshold ({upper} mg/dL)")
    ax.axhline(y=lower, color="green", linestyle="--", alpha=0.7, label=f"Hypo threshold ({lower} mg/dL)")

    ax.set_ylim(min(30, np.nanmin(cgm_timeseries.values)), max(np.nanmax(cgm_timeseries.values), 300))
    ax.set_xlabel("Time (hours)")
    time_grid_one_day = pd.date_range(start=start_time, periods=24, freq="1h")
    ax.set_xticks(time_grid_one_day)  # Show every hour from 0 to 24
    ax.set_xticklabels([f"{h.hour}" for h in time_grid_one_day])  # Format as HH:00
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    # plot the results
    return fig
