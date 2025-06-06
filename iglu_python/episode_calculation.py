from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import IGLU_R_COMPATIBLE, CGMS2DayByDay, check_data_columns, gd2d_to_df, get_local_tz


def episode_calculation(
    data: Union[pd.DataFrame, pd.Series],
    lv1_hypo: float = 70,
    lv2_hypo: float = 54,
    lv1_hyper: float = 180,
    lv2_hyper: float = 250,
    dur_length: int = 15,
    end_length: int = 15,
    return_data: bool = False,
    dt0: Optional[int] = None,
    inter_gap: int = 45,
    tz: str = "",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate Hypo/Hyperglycemic episodes with summary statistics.

    The function determines episodes or events, calculates summary statistics,
    and optionally returns data with episode label columns added.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values.
        Should only be data for 1 subject. In case multiple subject ids are detected,
        a warning is produced and only 1st subject is used.
    lv1_hypo : float, optional
        Level 1 hypoglycemia threshold. Default is 70 mg/dL.
    lv2_hypo : float, optional
        Level 2 hypoglycemia threshold. Default is 54 mg/dL.
    lv1_hyper : float, optional
        Level 1 hyperglycemia threshold. Default is 180 mg/dL.
    lv2_hyper : float, optional
        Level 2 hyperglycemia threshold. Default is 250 mg/dL.
    dur_length : int, optional
        Minimum duration in minutes to be considered an episode. Note dur_length should be
        a multiple of the data recording interval otherwise the function will round up to
        the nearest multiple. Default is 15 minutes to match consensus.
    end_length : int, optional
        Minimum duration in minutes of improved glycemia for an episode to end.
        Default is equal to dur_length to match consensus.
    return_data : bool, optional
        Whether to also return data with episode labels. Defaults to False which means
        only episode summary statistics will be returned.
    dt0 : Optional[float], optional
        The time frequency for interpolation in minutes, the default will match the
        CGM meter's frequency (e.g. 5 min for Dexcom).
    inter_gap : int, optional
        The maximum allowable gap (in minutes) for interpolation. The values will not
        be interpolated between the glucose measurements that are more than inter_gap
        minutes apart. The default value is 45 min.
    tz : str, optional
        A character string specifying the time zone to be used. System-specific,
        but "" is the current time zone, and "GMT" is UTC. Invalid values are most
        commonly treated as UTC, on some platforms with a warning.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        If return_data is False, a DataFrame with columns:
        - id: Subject id
        - type: Type of episode - either hypoglycemia or hyperglycemia
        - level: Level of episode - one of lv1, lv2, extended, lv1_excl
        - avg_ep_per_day: Average number of episodes per day
        - avg_ep_duration: Average duration of episodes in minutes
        - avg_ep_gl: Average glucose in the episode in mg/dL
        - total_episodes: Total number of episodes

        If return_data is True, returns a tuple where the first entry is the episode
        summary DataFrame (see above) and the second entry is the input data with
        episode labels added. Note the data returned here has been interpolated
        using the CGMS2DayByDay() function.

    Notes
    -----
    We follow the definition of episodes given in the 2023 consensus by Battelino et al.
    Note we have classified lv2 as a subset of lv1 since we find the consensus to be
    slightly ambiguous. For lv1 exclusive of lv2, please see lv1_excl which summarises
    episodes that were exclusively lv1 and did not cross the lv2 threshold. Also note,
    hypo extended refers to episodes that are >120 consecutive minutes below lv1 hypo
    and ends with at least 15 minutes of normoglycemia.

    References
    ----------
    Battelino et al. (2023): Continuous glucose monitoring and metrics for clinical
    trials: an international consensus statement
    Lancet Diabetes & Endocrinology 11(1) .42-57,
    doi:10.1016/s2213-8587(22)00319-9.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> episode_calculation(data)
       id    type    level  avg_ep_per_day  avg_ep_duration  avg_ep_gl  total_episodes
    0  subject1  hypo    lv1            X.XX            XX.X      XXX.X              X
    1  subject1  hypo    lv2            X.XX            XX.X      XXX.X              X
    ...
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        data = pd.DataFrame(
            {
                "id": ["subject1"] * len(data.values),
                "time": data.index,
                "gl": data.values,
            }
        )

    # Handle DataFrame input
    data = check_data_columns(data)

    # Check duration parameters
    if dur_length > inter_gap:
        print(
            "Warning: Interpolation gap parameter less than episode duration, "
            "data gaps may cause incorrect computation"
        )

    episode_data_df = pd.DataFrame(
        columns=[
            'id', 'time', 'gl', 'segment',
            'lv1_hypo', 'lv2_hypo', 'lv1_hyper', 'lv2_hyper',
            'ext_hypo', 'lv1_hypo_excl', 'lv1_hyper_excl'
        ]
    )
    episode_summary_df = pd.DataFrame(
        columns=[
            'id', 'type', 'level', 'avg_ep_per_day',
            'avg_ep_duration', 'avg_ep_gl', 'total_episodes'
        ]
    )

    # Process each subject ID separately
    for subject_id in data['id'].unique():
        # Get data for this subject
        subject_data = data[data['id'] == subject_id].copy()

        # Calculate episodes for this subject
        subject_summary, subject_episode_data = episode_single(
            subject_data,
            lv1_hypo=lv1_hypo,
            lv2_hypo=lv2_hypo,
            lv1_hyper=lv1_hyper,
            lv2_hyper=lv2_hyper,
            dur_length=dur_length,
            end_length=end_length,
            dt0=dt0,
            inter_gap=inter_gap,
            tz=tz,
        )

        subject_summary['id'] = subject_id
        subject_episode_data['id'] = subject_id

        # Append to main dataframes
        episode_data_df = pd.concat([episode_data_df, subject_episode_data], ignore_index=True)
        episode_summary_df = pd.concat([episode_summary_df, subject_summary], ignore_index=True)



    if return_data:
        return episode_summary_df, episode_data_df
    else:
        return episode_summary_df

def episode_single(
    data: pd.DataFrame,
    lv1_hypo: float,
    lv2_hypo: float,
    lv1_hyper: float,
    lv2_hyper: float,
    dur_length: int,
    end_length: int,
    dt0: Optional[float],
    inter_gap: int,
    tz: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify episodes for all segments for one subject.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    lv1_hypo : float
        Level 1 hypoglycemia threshold
    lv2_hypo : float
        Level 2 hypoglycemia threshold
    lv1_hyper : float
        Level 1 hyperglycemia threshold
    lv2_hyper : float
        Level 2 hyperglycemia threshold
    dur_length : int
        Minimum duration in minutes
    end_length : int
        Minimum duration in minutes for episode to end
    return_data : bool
        Whether to return labeled data
    dt0 : Optional[float]
        Time frequency in minutes
    inter_gap : int
        Maximum gap for interpolation
    tz : str
        Time zone

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        Either summary statistics or tuple of (summary, labeled data)
    """
    # Interpolate and segment data
    gd2d_tuple = CGMS2DayByDay(data, dt0=dt0, inter_gap=inter_gap, tz=tz)
    if dt0 is None:
        dt0 = gd2d_tuple[2]

    if IGLU_R_COMPATIBLE:
        day_one = pd.to_datetime(gd2d_tuple[1][0]).tz_localize(None) # make in naive-timezone
        day_one = day_one.tz_localize('UTC') # this is how IGLU_R works
        if tz and tz!="":
            day_one = day_one.tz_convert(tz)
        else:
            local_tz = get_local_tz()
            day_one = day_one.tz_convert(local_tz)
        ndays = len(gd2d_tuple[1])
        # generate grid times by starting from day one and cumulatively summing
        time_ip =  pd.date_range(start=day_one + pd.Timedelta(minutes=dt0), periods=ndays * 24 * 60 /dt0, freq=f"{dt0}min")
        data_ip = gd2d_tuple[0].flatten().tolist()
        new_data = pd.DataFrame({
            "time": time_ip,
            "gl": data_ip
            })
    else:
        new_data = gd2d_to_df(gd2d_tuple[0],gd2d_tuple[1],gd2d_tuple[2])


    # Check duration parameters
    if dur_length % dt0 != 0:
        print(
            "Warning: Episode duration is not a multiple of recording interval, "
            "smallest multiple that is greater than input dur_length chosen for computation"
        )

    dur_idx = int(np.ceil(dur_length / dt0))
    end_idx = int(np.ceil(end_length / dt0))

    # Step 1: Create boolean mask for NA values
    # R: na_idx = is.na(new_data$gl)
    na_idx = new_data['gl'].isna()

    # Step 2: Run-length encoding to find consecutive runs
    # R: segment_rle = rle(na_idx)$lengths
    segment_rle = _rle_lengths(na_idx)

    # Step 3: Copy data and create segment column
    # R: segment_data = new_data
    segment_data = new_data.copy()

    # Step 4: Create segment IDs by repeating segment numbers
    # R: segment_data$segment = rep(1:length(segment_rle), segment_rle)
    segment_ids = np.repeat(
        range(1, len(segment_rle) + 1),  # 1:length(segment_rle)
        segment_rle                      # repeat counts
    )
    segment_data['segment'] = segment_ids

    # Step 5: Remove rows with NA glucose values
    # R: segment_data = segment_data[!is.na(segment_data$gl), ]
    segment_data = segment_data[~segment_data['gl'].isna()].reset_index(drop=True)


    # Classify events for each segment
    ep_per_seg = (
        segment_data.groupby("segment")
        .apply(
            lambda x: pd.DataFrame(
                {
                    "lv1_hypo": event_class(x, "hypo", lv1_hypo, dur_idx, end_idx),
                    "lv2_hypo": event_class(x, "hypo", lv2_hypo, dur_idx, end_idx),
                    "lv1_hyper": event_class(x, "hyper", lv1_hyper, dur_idx, end_idx),
                    "lv2_hyper": event_class(x, "hyper", lv2_hyper, dur_idx, end_idx),
                    "ext_hypo": event_class(
                        x, "hypo", lv1_hypo, int(120 / dt0) + 1, end_idx
                    ),
                }
            )
        )
        .reset_index()
        .drop(columns=['level_1'])
    )


    # Add exclusive labels
    def hypo_exclusion_logic(group_df):
        # group_df is a DataFrame with all columns for the current group
        if (group_df['lv2_hypo'] > 0).any():
            return pd.Series([0] * len(group_df), index=group_df.index)
        else:
            return group_df['lv1_hypo']
    ep_per_seg['lv1_hypo_excl'] = ep_per_seg.groupby(['segment', 'lv1_hypo']).apply(hypo_exclusion_logic).reset_index(level=[0,1], drop=True).values.flatten()

    def hyper_exclusion_logic(group_df):
        # group_df is a DataFrame with all columns for the current group
        if (group_df['lv2_hyper'] > 0).any():
            return pd.Series([0] * len(group_df), index=group_df.index)
        else:
            return group_df['lv1_hyper']
    ep_per_seg['lv1_hyper_excl'] = ep_per_seg.groupby(['segment', 'lv1_hyper']).apply(hyper_exclusion_logic).reset_index(level=[0,1], drop=True).values.flatten()

    full_segment_df = pd.concat([segment_data, ep_per_seg.drop(["segment"], axis=1)], axis=1)

    # Calculate summary statistics
    summary_df = episode_summary(full_segment_df, dt0)
    return summary_df, full_segment_df

def event_class(
    data: pd.DataFrame,
    level_type: str,
    threshold: float,
    event_duration: int,
    end_duration: int,
) -> np.ndarray:
    """
    Classify and label all events in a segment.

    ### algorithm description
    # (1) apply the following to each subject (episode_single)
    #       (a) interpolate to create equidistant grid
    #       (b) split into contiguous segments based on gaps
    #       (c) classify events in each segment (event_class)
    #       (d) summarize episodes (episode_summary)
    # (a) event_class: label events of each type for each segment
    #       (a) must be >= duration (function input is # idx to get # minutes)
    #       (b) ends at >= dur_length (function input is top level dur_length/dt0)
    # (b) episode_summary: calculate summary statistics
    #       (a) return for each type of episode: # episodes, mean duration, mean glu value

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', and 'gl'
    level_type : str
        Either 'hypo' or 'hyper' to indicate type of event
    threshold : float
        Glucose threshold for event classification
    event_duration : int
        Minimum duration in indices to be considered an event
    end_duration : int
        Minimum duration in indices for event to end

    Returns
    -------
    np.ndarray
        Array of event labels, 0 for no event, positive integer for event
    """
    # Create annotated dataframe
    annotated = data.copy()

    # Mark events based on level type
    if level_type == "hypo":
        annotated["level"] = annotated["gl"] < threshold
    elif level_type == "hyper":
        annotated["level"] = annotated["gl"] > threshold

    # Get run lengths of events
    level_rle = _rle_lengths(annotated["level"])
    annotated["event"] = np.repeat(np.arange(1, len(level_rle) + 1), level_rle)

    # Group by event and calculate start/end positions
    annotated_grouped = (
        annotated.groupby("event")
        .apply(
            lambda x: pd.DataFrame(
                {
                    # possibly event; where duration is met
                    "pos_start": [x["level"].iloc[0] and (len(x) >= event_duration)]*len(x),
                    # if possible event, add start on first index of event
                    "start": (
                        ["start"
                        if (x["level"].iloc[0] and len(x) >= event_duration)
                        else None] + [None]*(len(x)-1)
                    ),
                    # add possible ends (always need to check for end duration)
                    "pos_end": [not x["level"].iloc[0] and (len(x) >= end_duration)]*len(x),
                    "end": (
                        ["end"
                        if (not x["level"].iloc[0] and len(x) >= end_duration)
                        else None] + [None]*(len(x)-1)
                    ),
                }
            )
        )
        .reset_index()
        .drop(columns=['level_1'])
    )

    annotated = pd.concat([annotated,annotated_grouped.drop(["event"], axis=1)], axis=1)

    ### for each possible end find the matching start
    # Get start and end positions
    starts = annotated[annotated["start"] == "start"].index.tolist()
    ends = [0] + annotated[annotated["end"] == "end"].index.tolist() + [len(data)]

    # If no episodes, return zeros
    if not starts:
        return np.zeros(len(data))

    # Find matching pairs
    pairs = pd.DataFrame(
        {
            "starts_ref": starts,
            "ends_ref": [min([e for e in ends if e > s]) - 1 for s in starts],
        }
    )

    # Remove intervening starts
    pairs = pairs.drop_duplicates(subset=["ends_ref"])

    # Create event labels
    event_idx = []
    event_label = []
    for i, (start, end) in enumerate(zip(pairs["starts_ref"], pairs["ends_ref"], strict=False)):
        event_idx.extend(range(start, end + 1))
        event_label.extend([i + 1] * (end - start + 1))

    # Create output array
    output = np.zeros(len(data))
    output[event_idx] = event_label

    return output


def lv1_excl(data: pd.DataFrame) -> np.ndarray:
    """
    Label exclusive level 1 events (1 vs 0 if not).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'id', 'time', 'gl', 'segment', 'lv1', 'lv2'

    Returns
    -------
    np.ndarray
        Array of exclusive level 1 event labels
    """
    # all lv1 columns
    lv1 = [column for column in data.columns if column.startswith("lv1")]
    lv1_first = lv1[0]
    lv2 = [column for column in data.columns if column.startswith("lv2")]
    lv2_first = lv2[0]
    # Group by segment and lv1
    grouped = data.groupby(["segment", lv1_first])

    # Calculate exclusive labels
    excl = grouped.apply(
        lambda x: pd.DataFrame(
                {
                    "excl":[0 if (x[lv2_first].values > 0).any() else x[lv1_first].iloc[0]]*len(x)
                })
    )

    excl = excl.reset_index()

    return excl[['segment','excl']]


def episode_summary(data: pd.DataFrame, dt0: float) -> pd.DataFrame:
    """
    Calculate summary statistics for each type of episode.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with episode labels
    dt0 : float
        Time frequency in minutes

    Returns
    -------
    pd.DataFrame
        Summary statistics for each episode type
    """

    def episode_summary_helper(
        data: pd.DataFrame, level_label: str, dt0: float
    ) -> List[float]:
        """Helper function to calculate summary for one episode type"""
        # Select relevant columns
        data = data[[ "time", "gl", "segment", level_label]].copy()
        data.columns = ["time", "gl", "segment", "event"]

        # If no events, return zeros/NA
        if all(data["event"] == 0):
            return [0, 0, np.nan, 0]

        # Calculate summary metrics
        events = data[data["event"] != 0][["gl", "segment", "event"]]
        data_sum = (
            events.groupby(["segment", "event"])
            .agg({"gl": ["count", "mean"]})
            .reset_index()
        )

        # Calculate metrics
        avg_ep_per_day = len(data_sum) / (len(data) * dt0 / 60 / 24)
        avg_ep_duration = data_sum[("gl", "count")].mean() * dt0
        avg_ep_gl = data_sum[("gl", "mean")].mean()
        total_episodes = len(data_sum)

        return [avg_ep_per_day, avg_ep_duration, avg_ep_gl, total_episodes]

    # Calculate summaries for each episode type
    labels = [
        "lv1_hypo",
        "lv2_hypo",
        "ext_hypo",
        "lv1_hyper",
        "lv2_hyper",
        "lv1_hypo_excl",
        "lv1_hyper_excl",
    ]
    out_list = [episode_summary_helper(data, label, dt0) for label in labels]

    # Create output DataFrame
    output = pd.DataFrame(
        {
            "type": ["hypo"] * 3 + ["hyper"] * 2 + ["hypo", "hyper"],
            "level": ["lv1", "lv2", "extended", "lv1", "lv2", "lv1_excl", "lv1_excl"],
            "avg_ep_per_day": [x[0] for x in out_list],
            "avg_ep_duration": [x[1] for x in out_list],
            "avg_ep_gl": [x[2] for x in out_list],
            "total_episodes": [x[3] for x in out_list],
        }
    )

    return output

def _rle_lengths(boolean_series):
    """Python equivalent of R's rle()$lengths"""
    # Find where values change
    changes = boolean_series != boolean_series.shift(1)
    # Group by change points and get group sizes
    groups = changes.cumsum()
    return boolean_series.groupby(groups).size().values
