"""
This module implements various plots for the iglu_python package.
"""

import matplotlib.pyplot as plt
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
        axes[i].set_ylim(0, max(max(cgm_one_day.values), 300))

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

    return fig
