import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional
from .utils import check_data_columns, CGMS2DayByDay

def event_class(data: pd.DataFrame, level_type: str, threshold: float, 
                event_duration: int, end_duration: int) -> np.ndarray:
    """
    Classify and label all events in a segment.
    
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
    if level_type == 'hypo':
        annotated['level'] = annotated['gl'] < threshold
    elif level_type == 'hyper':
        annotated['level'] = annotated['gl'] > threshold
    
    # Get run lengths of events
    level_rle = np.array([len(list(g)) for _, g in pd.groupby(annotated['level'])])
    annotated['event'] = np.repeat(np.arange(1, len(level_rle) + 1), level_rle)
    
    # Group by event and calculate start/end positions
    annotated = annotated.groupby('event').apply(
        lambda x: pd.Series({
            'pos_start': x['level'].iloc[0] and (len(x) >= event_duration),
            'start': 'start' if (x['level'].iloc[0] and len(x) >= event_duration) else None,
            'pos_end': not x['level'].iloc[0] and (len(x) >= end_duration),
            'end': 'end' if (not x['level'].iloc[0] and len(x) >= end_duration) else None
        })
    ).reset_index()
    
    # Get start and end positions
    starts = annotated[annotated['start'] == 'start'].index.tolist()
    ends = [0] + annotated[annotated['end'] == 'end'].index.tolist() + [len(data)]
    
    # If no episodes, return zeros
    if not starts:
        return np.zeros(len(data))
    
    # Find matching pairs
    pairs = pd.DataFrame({
        'starts_ref': starts,
        'ends_ref': [min([e for e in ends if e > s]) - 1 for s in starts]
    })
    
    # Remove intervening starts
    pairs = pairs.drop_duplicates(subset=['ends_ref'])
    
    # Create event labels
    event_idx = []
    event_label = []
    for i, (start, end) in enumerate(zip(pairs['starts_ref'], pairs['ends_ref'])):
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
    # Group by segment and lv1
    grouped = data.groupby(['segment', 'lv1'])
    
    # Calculate exclusive labels
    excl = grouped.apply(
        lambda x: 0 if any(x['lv2'] > 0) else x['lv1'].iloc[0]
    ).reset_index()
    
    return excl[0].values

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
    def episode_summary_helper(data: pd.DataFrame, level_label: str, dt0: float) -> List[float]:
        """Helper function to calculate summary for one episode type"""
        # Select relevant columns
        data = data[['id', 'time', 'gl', 'segment', level_label]].copy()
        data.columns = ['id', 'time', 'gl', 'segment', 'event']
        
        # If no events, return zeros/NA
        if all(data['event'] == 0):
            return [0, 0, np.nan, 0]
        
        # Calculate summary metrics
        events = data[data['event'] != 0][['gl', 'segment', 'event']]
        data_sum = events.groupby(['segment', 'event']).agg({
            'gl': ['count', 'mean']
        }).reset_index()
        
        # Calculate metrics
        avg_ep_per_day = len(data_sum) / (len(data) * dt0 / 60 / 24)
        avg_ep_duration = data_sum[('gl', 'count')].mean() * dt0
        avg_ep_gl = data_sum[('gl', 'mean')].mean()
        total_episodes = len(data_sum)
        
        return [avg_ep_per_day, avg_ep_duration, avg_ep_gl, total_episodes]
    
    # Calculate summaries for each episode type
    labels = ['lv1_hypo', 'lv2_hypo', 'ext_hypo', 'lv1_hyper', 'lv2_hyper',
              'lv1_hypo_excl', 'lv1_hyper_excl']
    out_list = [episode_summary_helper(data, label, dt0) for label in labels]
    
    # Create output DataFrame
    output = pd.DataFrame({
        'type': ['hypo'] * 3 + ['hyper'] * 2 + ['hypo', 'hyper'],
        'level': ['lv1', 'lv2', 'extended', 'lv1', 'lv2', 'lv1_excl', 'lv1_excl'],
        'avg_ep_per_day': [x[0] for x in out_list],
        'avg_ep_duration': [x[1] for x in out_list],
        'avg_ep_gl': [x[2] for x in out_list],
        'total_episodes': [x[3] for x in out_list]
    })
    
    return output

def episode_single(data: pd.DataFrame, lv1_hypo: float, lv2_hypo: float,
                  lv1_hyper: float, lv2_hyper: float, dur_length: int,
                  end_length: int, return_data: bool, dt0: Optional[float],
                  inter_gap: int, tz: str) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
    data_ip = CGMS2DayByDay(data, dt0=dt0, inter_gap=inter_gap, tz=tz)
    dt0 = data_ip[2]
    
    # Generate interpolated times
    day_one = pd.Timestamp(data_ip[1][0], tz=tz)
    ndays = len(data_ip[1])
    time_ip = day_one + pd.Timedelta(minutes=dt0) * np.arange(ndays * 24 * 60 / dt0)
    
    # Create interpolated DataFrame
    new_data = pd.DataFrame({
        'id': data['id'].iloc[0],
        'time': time_ip,
        'gl': data_ip[0].flatten()
    })
    
    # Check duration parameters
    if dur_length % dt0 != 0:
        print("Warning: Episode duration is not a multiple of recording interval, "
              "smallest multiple that is greater than input dur_length chosen for computation")
    
    dur_idx = int(np.ceil(dur_length / dt0))
    end_idx = int(np.ceil(end_length / dt0))
    
    # Create segments based on gaps
    na_idx = new_data['gl'].isna()
    segment_rle = [len(list(g)) for _, g in pd.groupby(na_idx)]
    new_data['segment'] = np.repeat(np.arange(1, len(segment_rle) + 1), segment_rle)
    segment_data = new_data[~new_data['gl'].isna()].copy()
    
    # Classify events for each segment
    ep_per_seg = segment_data.groupby('segment').apply(
        lambda x: pd.DataFrame({
            'lv1_hypo': event_class(x, 'hypo', lv1_hypo, dur_idx, end_idx),
            'lv2_hypo': event_class(x, 'hypo', lv2_hypo, dur_idx, end_idx),
            'lv1_hyper': event_class(x, 'hyper', lv1_hyper, dur_idx, end_idx),
            'lv2_hyper': event_class(x, 'hyper', lv2_hyper, dur_idx, end_idx),
            'ext_hypo': event_class(x, 'hypo', lv1_hypo, int(120/dt0) + 1, end_idx)
        })
    ).reset_index()
    
    # Add exclusive labels
    ep_per_seg['lv1_hypo_excl'] = lv1_excl(pd.concat([
        segment_data[['id', 'time', 'gl', 'segment']],
        ep_per_seg[['lv1_hypo', 'lv2_hypo']]
    ], axis=1))
    ep_per_seg['lv1_hyper_excl'] = lv1_excl(pd.concat([
        segment_data[['id', 'time', 'gl', 'segment']],
        ep_per_seg[['lv1_hyper', 'lv2_hyper']]
    ], axis=1))
    
    # Return data if requested
    if return_data:
        return pd.concat([segment_data, ep_per_seg.drop('segment', axis=1)], axis=1)
    
    # Calculate summary statistics
    return episode_summary(pd.concat([segment_data, ep_per_seg.drop('segment', axis=1)], axis=1), dt0)

def episode_calculation(data: Union[pd.DataFrame, pd.Series],
                       lv1_hypo: float = 70, lv2_hypo: float = 54,
                       lv1_hyper: float = 180, lv2_hyper: float = 250,
                       dur_length: int = 15, end_length: int = 15,
                       return_data: bool = False, dt0: Optional[float] = None,
                       inter_gap: int = 45, tz: str = "") -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
        data = pd.DataFrame({
            'id': ['subject1'],
            'time': pd.date_range(start='2020-01-01', periods=len(data), freq='5min'),
            'gl': data
        })
    
    # Handle DataFrame input
    data = check_data_columns(data)
    
    # Check duration parameters
    if dur_length > inter_gap:
        print("Warning: Interpolation gap parameter less than episode duration, "
              "data gaps may cause incorrect computation")
    
    # Calculate episodes for each subject
    out = data.groupby('id').apply(
        lambda x: episode_single(
            x, lv1_hypo=lv1_hypo, lv2_hypo=lv2_hypo,
            lv1_hyper=lv1_hyper, lv2_hyper=lv2_hyper,
            return_data=False, dur_length=dur_length,
            end_length=end_length, dt0=dt0,
            inter_gap=inter_gap, tz=tz
        )
    ).reset_index()
    
    if return_data:
        ep_data = data.groupby('id').apply(
            lambda x: episode_single(
                x, lv1_hypo=lv1_hypo, lv2_hypo=lv2_hypo,
                lv1_hyper=lv1_hyper, lv2_hyper=lv2_hyper,
                return_data=True, dur_length=dur_length,
                end_length=end_length, dt0=dt0,
                inter_gap=inter_gap, tz=tz
            )
        ).reset_index()
        return {'episodes': out, 'data': ep_data}
    
    return out 