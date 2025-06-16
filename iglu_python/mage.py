from typing import Literal, Union

import numpy as np
import pandas as pd

from .utils import CGMS2DayByDay, check_data_columns


def mage(
    data: Union[pd.DataFrame, pd.Series],
    version: Literal["ma", "naive"] = "ma",
    sd_multiplier: float = 1.0,
    short_ma: int = 5,
    long_ma: int = 32,
    return_type: Literal["num", "df"] = "num",
    direction: Literal["avg", "service", "max", "plus", "minus"] = "avg",
    tz: str = "",
    inter_gap: int = 45,
    max_gap: int = 180,
) -> pd.DataFrame:
    """
    Calculate Mean Amplitude of Glycemic Excursions (MAGE).

    The function calculates MAGE values using either a moving average ('ma') or naive ('naive') algorithm.
    The 'ma' algorithm is more accurate and is the default. It uses crosses of short and long moving
    averages to identify intervals where a peak/nadir might exist, then calculates the height from
    one peak/nadir to the next nadir/peak from the original glucose values.

    If version 'ma' is selected, the function computationally emulates the manual method for calculating
    the mean amplitude of glycemic excursions (MAGE) first suggested in
    "Mean Amplitude of Glycemic Excursions, a Measure of Diabetic Instability", (Service, 1970).
    For this version, glucose values will be interpolated over a uniform time grid prior to calculation.

    'ma' is a more accurate algorithm that uses the crosses of a short and long moving average
    to identify intervals where a peak/nadir might exist. Then, the height from one peak/nadir
    to the next nadir/peak is calculated from the _original_ (not moving average) glucose values.
    (Note: this function internally uses CGMS2DayByDay with dt0 = 5.
    Thus, all CGM data is linearly interpolated to 5 minute intervals. See the MAGE vignette for more details.)

    'naive' algorithm calculates MAGE by taking the mean of absolute glucose differences
    (between each value and the mean)  that are greater than the standard deviation. A multiplier can be added
    to the standard deviation using the `sd_multiplier` argument.


    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        DataFrame with columns 'id', 'time', and 'gl', or a Series of glucose values
    version : Literal['ma', 'naive'], default='ma'
        Algorithm version to use. 'ma' is more accurate and is the default.
        'naive' is included for backward compatibility.
    sd_multiplier : float, default=1.0
        Multiplier for standard deviation used in naive algorithm to determine
        size of glycemic excursions. Only used if version='naive'.
    short_ma : int, default=5
        Period length of short moving average. Must be positive and less than long_ma.
        Recommended < 15.
    long_ma : int, default=32
        Period length of long moving average. Must be positive and greater than short_ma.
        Recommended > 20.
    return_type : Literal['num', 'df'], default='num'
        Return type. 'num' returns a single MAGE value, 'df' returns a DataFrame with
        MAGE values for each segment.
    direction : Literal['avg', 'service', 'max', 'plus', 'minus'], default='avg'
        Direction of MAGE calculation:
        - 'avg': Average of MAGE+ and MAGE-
        - 'service': Based on first countable excursion
        - 'max': Maximum of MAGE+ and MAGE-
        - 'plus': MAGE+ (nadir to peak)
        - 'minus': MAGE- (peak to nadir)
    tz : str, default=""
        Time zone to use for datetime conversion. Empty string means use local time zone.
    inter_gap : int, default=45
        Maximum gap in minutes for interpolation. Gaps larger than this will not be
        interpolated.
    max_gap : int, default=180
        Maximum gap in minutes before splitting into segments.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: subject identifier (if DataFrame input)
        - MAGE: Mean Amplitude of Glycemic Excursions value

    References
    ----------
    Service et al. (1970) Mean amplitude of glycemic excursions, a measure of diabetic instability
    Diabetes 19:644-655, doi:10.2337/diab.19.9.644.

    Fernandes, Nathaniel J., et al. "Open-source algorithm to calculate mean amplitude of glycemic
    excursions using short and long moving averages." Journal of diabetes science and technology
    16.2 (2022): 576-577, doi:10.1177/19322968211061165.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': ['subject1', 'subject1', 'subject2', 'subject2'],
    ...     'time': ['2020-01-01 00:00:00', '2020-01-01 00:05:00',
    ...              '2020-01-01 00:00:00', '2020-01-01 00:05:00'],
    ...     'gl': [150, 200, 130, 190]
    ... })
    >>> data['time'] = pd.to_datetime(data['time'])
    >>> mage(data)
       id    MAGE
    0  subject1  50.0
    1  subject2  60.0

    >>> mage(data['gl'], version='naive', sd_multiplier=1.5)
       MAGE
    0  45.0
    """
    # -------------------
    # start mage()
    # Handle Series input
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        data_df = pd.DataFrame(
            {
                "id": ["subject1"] * len(data.values),
                "time": data.index,
                "gl": data.values,
            }
        )
        if version == "ma":
            mage_val = mage_ma_single(data_df, short_ma, long_ma, direction,
                                      return_type='num',
                                      inter_gap=inter_gap,
                                      max_gap=max_gap,
                                      tz=tz)
        else:
            mage_val = mage_naive(data_df,sd_multiplier=sd_multiplier)
        return mage_val

    # Handle DataFrame input
    data = check_data_columns(data)

    # Calculate MAGE for each subject
    result = []
    for subject in data["id"].unique():
        subject_data = data[data["id"] == subject].copy()
        if len(subject_data.dropna(subset=["gl"])) == 0:
            continue

        if version == "ma":
            mage_val = mage_ma_single(subject_data, short_ma, long_ma, direction, return_type,inter_gap,max_gap,tz)
            if return_type == "df" :
                subject_result_dict = mage_val.to_dict()
            else:
                subject_result_dict = {"MAGE": mage_val}
        else:
            mage_val = mage_naive(subject_data,sd_multiplier=sd_multiplier)
            subject_result_dict = {"MAGE": mage_val}

        result.append({"id": subject, **subject_result_dict})

    return pd.DataFrame(result)

def mage_naive(data: pd.DataFrame,sd_multiplier:float = 1.0) -> float:
    """Calculate MAGE using naive algorithm"""
    # Calculate absolute differences from mean
    mean_gl = data["gl"].mean()
    abs_diff_mean = abs(data["gl"] - mean_gl)

    # Calculate standard deviation
    std_gl = data["gl"].std()

    # Calculate MAGE as mean of differences greater than sd_multiplier * std
    mage_val = abs_diff_mean[abs_diff_mean > (sd_multiplier * std_gl)].mean()

    return float(mage_val) if not pd.isna(mage_val) else np.nan

def mage_ma_single(data: pd.DataFrame, short_ma: int, long_ma: int,
        direction:str ='avg',
        return_type:str = "num",
        inter_gap:int = 45,
        max_gap:int = 180,
        tz:str = "" ) -> pd.DataFrame|float:
    """Calculate MAGE using moving average algorithm for a single subject"""
    ## 1. Preprocessing
    # 1.1 Interpolate over uniform grid
    # Note: always interpolate to 5 minute grid
    data_ip = CGMS2DayByDay(data, dt0=5, inter_gap=inter_gap, tz=tz)
    data_ip[2]  # Time between measurements in minutes
    # replace for 5 min to fix bug in CGMS2DayByDay
    day_one = data_ip[1][0]
    len(data_ip[1])

    # 1.2 Generate grid times by starting from day one and cumulatively summing
    # note fix 5 min used in interpretation
    gl = data_ip[0].flatten().tolist()
    time_ip = [pd.Timedelta(i * 5, unit="m") + day_one for i in range(1,len(gl)+1)]

    # 1.3 Recalculate short_ma and long_ma because short and long are based on 5 minutes originally
    # > Multiply by 5 to get length in min
    # > Divide by dt0 to get rounded number of measurements that are roughly equal to original short/long ma definition
    # short_ma = round(short_ma*5/dt0)
    # long_ma = round(long_ma*5/dt0)
    # Ensure short_ma and long_ma are appropriate
    if short_ma >= long_ma:
        short_ma, long_ma = long_ma, short_ma

    ## 2. Change to interpolated data (times and glucose)
    # > change data into id, interpolated times, interpolated glucose (t to get rowwise)
    # > drop NA rows before first glucose reading
    # > then drop NA rows after last glucose reading
    # > Label NA glucose as gap (gap = 1)
    interpolated_data = pd.DataFrame({
        "id" : data['id'].iloc[0],
        "time": pd.Series(time_ip, dtype='datetime64[ns]'),
        "gl": pd.Series(gl, dtype='float64')
    })
    # Drop NA rows before first glucose reading
    first_valid_idx = interpolated_data['gl'].first_valid_index()
    if first_valid_idx is not None:
        interpolated_data = interpolated_data.iloc[first_valid_idx:]
    # Drop NA rows after last glucose reading
    last_valid_idx = interpolated_data['gl'].last_valid_index()
    if last_valid_idx is not None:
        interpolated_data = interpolated_data.iloc[:last_valid_idx+1]
    # Add gap column to mark NA values as 1
    interpolated_data['gap'] = interpolated_data['gl'].isna().astype(int)

    # 4. Time Series Segmentation: split gaps > max_gap into separate segments
    dfs = segment_time_series(interpolated_data,max_gap)  # note: max_gap is in minutes

    # 5. Calculate MAGE on each identified segment
    return_val = pd.DataFrame(columns=["start", "end", "mage", "plus_or_minus", "first_excursion"])
    for segment in dfs:
        ret = mage_atomic(segment,short_ma,long_ma)
        if return_val.empty:
            return_val = ret
        else:
            return_val = pd.concat([return_val, ret], ignore_index=True)

    if return_type == 'df':
        return return_val

    """Process MAGE results with filtering and weighting."""
    # Filter by direction (equivalent to the previous R filtering code)
    if direction == 'plus':
        res = return_val[return_val['plus_or_minus'] == 'PLUS'].copy()
    elif direction == 'minus':
        res = return_val[return_val['plus_or_minus'] == 'MINUS'].copy()
    elif direction == 'avg':
        res = return_val[return_val['MAGE'].notna()].copy()
    elif direction == 'max':
        # Group by start,end and keep max mage in each group
        idx = return_val.groupby(['start', 'end'])['MAGE'].idxmax()
        res = return_val.loc[idx].reset_index(drop=True)
    else:  # default: first excursions only
        res = return_val[return_val['first_excursion']].copy()

    # Calculate time-weighted MAGE
    if res.empty:
        return np.nan

    res['hours'] = res['end'] - res['start']
    res['weight'] = res['hours'] / res['hours'].sum()
    weighted_mage = (res['MAGE'] * res['weight']).sum()

    return weighted_mage

def mage_atomic(data, short_ma,long_ma):
    """ 0. Calculates MAGE on 1 segment of CGM trace """

    # 2c. Calculate the moving average values
    data = data.copy()
    data["MA_Short"] = data["gl"].rolling(window=short_ma, min_periods=1).mean()
    data["MA_Long"] = data["gl"].rolling(window=long_ma, min_periods=1).mean()
    # Fill leading NAs (forward fill first valid value)
    if short_ma > len(data):
        data.loc[data.index[:short_ma], 'MA_Short'] = data['MA_Short'].iloc[-1]
    else:
        data.loc[data.index[:short_ma], 'MA_Short'] = data['MA_Short'].iloc[short_ma-1]
    if long_ma > len(data):
        data.loc[data.index[:long_ma], 'MA_Long'] = data['MA_Long'].iloc[-1]
    else:
        data.loc[data.index[:long_ma], 'MA_Long'] = data['MA_Long'].iloc[long_ma-1]
    # Calculate difference
    data['DELTA_SHORT_LONG'] = data['MA_Short'] - data['MA_Long']
    data = data.reset_index(drop=True)
    nmeasurements = len(data)

    # Sanity check
    if (
        data['gl'].isnull().all() or
        nmeasurements < 7 or
        nmeasurements < short_ma or
        np.std(data['gl'], ddof=1) < 1
    ):
        return pd.DataFrame({
            'start': [data['time'].iloc[0]],
            'end': [data['time'].iloc[-1]],
            'MAGE': [np.nan],
            'plus_or_minus': [np.nan],
            'first_excursion': [np.nan]
        })

    # 2d. Create a preallocated list of crossing point ids & type
    # Find crossing points
    # Detect trend reversal points in glucose data using DELTA signal.
    # Initialize variables
    idx = list(data.index)  # R: idx = as.numeric(rownames(.data))
    types = {'REL_MIN': 0, 'REL_MAX': 1}  # R: types = list2env(list(REL_MIN=0, REL_MAX=1))

    # Create storage lists - R: list_cross <- list("id" = rep.int(NA, nmeasurements),
    # "type" = rep.int(NA, nmeasurements))
    list_cross = {
        'id': [np.nan] * nmeasurements,
        'type': [np.nan] * nmeasurements
    }

    # Always add 1st point
    list_cross['id'][0] = idx[0]
    list_cross['type'][0] = types['REL_MAX'] if data['DELTA_SHORT_LONG'].iloc[0] > 0 else types['REL_MIN']
    count = 1  # Python uses 0-based indexing, so count starts at 1

    # treat DELTA_SHORT_LONG==0 as NaN ( so we can skip its multiplication)
    data.loc[data['DELTA_SHORT_LONG'] == 0, 'DELTA_SHORT_LONG'] = np.nan

    # Main loop - R: for(i in 2:length(.data$DELTA_SHORT_LONG))
    for i in range(1, len(data['DELTA_SHORT_LONG'])):
        # Check data validity
        if (not pd.isna(data['gl'].iloc[i]) and
            not pd.isna(data['gl'].iloc[i-1]) and
            not pd.isna(data['DELTA_SHORT_LONG'].iloc[i]) and
            not pd.isna(data['DELTA_SHORT_LONG'].iloc[i-1])):

            # Primary crossover detection: crossing point if DELTA changes sign
            if (data['DELTA_SHORT_LONG'].iloc[i] * data['DELTA_SHORT_LONG'].iloc[i-1] < 0):
                list_cross['id'][count] = idx[i]
                if data['DELTA_SHORT_LONG'].iloc[i] < data['DELTA_SHORT_LONG'].iloc[i-1]:
                    list_cross['type'][count] = types['REL_MIN']
                else:
                    list_cross['type'][count] = types['REL_MAX']
                count += 1

        # Gap handling: needed for gaps, where DELTA_SHORT_LONG(i-1 | i-2) = NaN
        elif (not pd.isna(data['DELTA_SHORT_LONG'].iloc[i]) and
            count >= 1):  # Make sure we have a previous crossover

            # R: match(list_cross$id[count-1], idx) - find index of previous crossover
            try:
                prev_cross_idx = idx.index(list_cross['id'][count-1])
                prev_delta = data['DELTA_SHORT_LONG'].iloc[prev_cross_idx]

                if (data['DELTA_SHORT_LONG'].iloc[i] * prev_delta < 0):
                    list_cross['id'][count] = idx[i]
                    if data['DELTA_SHORT_LONG'].iloc[i] < prev_delta:
                        list_cross['type'][count] = types['REL_MIN']
                    else:
                        list_cross['type'][count] = types['REL_MAX']
                    count += 1
            except ValueError:
                # Handle case where previous crossover id not found in idx
                pass

    # Add last point to capture excursion at end
    # R: utils::tail(idx, 1)
    last_idx = idx[-1]
    list_cross['id'][count] = last_idx

    if data['DELTA_SHORT_LONG'].iloc[-1] > 0:
        list_cross['type'][count] = types['REL_MAX']
    else:
        list_cross['type'][count] = types['REL_MIN']

    # Filter out NaN values - R: list_cross$id[!is.na(list_cross$id)]
    clean_ids = [x for x in list_cross['id'] if not pd.isna(x)]
    clean_types = [x for x in list_cross['type'] if not pd.isna(x)]

    # Create DataFrame - R: do.call(cbind.data.frame, list_cross)
    crosses = pd.DataFrame({
            "id":clean_ids,
            "type":clean_types
        })

    # 2e. Calculate min and max glucose values from ids and types in crosses + store indexes for plotting later
    # R: num_extrema = nrow(crosses)-1
    num_extrema = len(crosses) - 1

    # R: minmax <- rep(NA_real_, num_extrema), indexes <- rep(NA_real_, num_extrema)
    minmax = [np.nan] * num_extrema
    indexes = [np.nan] * num_extrema

    # R: for(i in 1:num_extrema)
    for i in range(num_extrema):
        # Define search boundaries
        # R: s1 <- ifelse(i == 1, crosses[i, 1], indexes[i-1])
        if i == 0:  # First extrema
            s1 = int(crosses.iloc[i]['id'])  # crosses[i, 1] in R (1-indexed)
        else:
            s1 = int(indexes[i-1])  # last minmax index

        # R: s2 <- crosses[i+1,1]
        s2 = int(crosses.iloc[i+1]['id'])  # crosses[i+1, 1] in R

        # Extract glucose segment - R: .data[as.character(s1:s2), ]$gl
        segment_start = s1
        segment_end = s2
        glucose_segment = data['gl'].iloc[segment_start:segment_end+1] # including next cross point

        # Find min or max based on crossover type
        if crosses.iloc[i]['type'] == types['REL_MIN']:  # crosses[i, "type"] in R
            # R: min(.data[as.character(s1:s2), ]$gl, na.rm = TRUE)
            minmax[i] = glucose_segment.min()
            # R: which.min(.data[as.character(s1:s2), ]$gl)+s1-1
            indexes[i] = glucose_segment.idxmin()
        else:
            # R: max(.data[as.character(s1:s2), ]$gl, na.rm = TRUE)
            minmax[i] = glucose_segment.max()
            # R: which.max(.data[as.character(s1:s2), ]$gl)+s1-1
            indexes[i] = glucose_segment.idxmax()

    # excursion elimination
    differences = np.subtract.outer(minmax, minmax).T
    standardD = data['gl'].std()  # pandas uses sample std dev by default
    len(minmax)


    # MAGE+ algorithm, which identifies and measures positive glycemic excursions
    # (nadir-to-peak movements that exceed the standard deviation threshold).
    mage_plus_heights, mage_plus_tp_pairs = calculate_mage_plus(differences, minmax, standardD)

    # MAGE- algorithm, which identifies and measures negative glycemic excursions
    # (peak-to-nadir movements that exceed the standard deviation threshold).
    mage_minus_heights, mage_minus_tp_pairs = calculate_mage_minus(differences, minmax, standardD)

    if len(mage_minus_heights) == 0 and len(mage_plus_heights) == 0:
        return pd.DataFrame({
            'start': [data['time'].iloc[0]],
            'end': [data['time'].iloc[-1]],
            'MAGE': [np.nan],
            'plus_or_minus': [np.nan],
            'first_excursion': [np.nan]
        }, index=[0])

    # Determine which excursion type occurs first
    if (len(mage_plus_heights) > 0 and
        (len(mage_minus_heights) == 0 or
        mage_plus_tp_pairs[0][1] <= mage_minus_tp_pairs[0][0])):
        is_plus_first = True
    else:
        is_plus_first = False

    # Create MAGE+ result dataframe
    mage_plus = pd.DataFrame({
        'start': [data['time'].iloc[0]],
        'end': [data['time'].iloc[-1]],
        'MAGE': [np.mean(mage_plus_heights) if len(mage_plus_heights) > 0 else np.nan],
        'plus_or_minus': ['PLUS'],
        'first_excursion': [is_plus_first]
    })

    # Create MAGE- result dataframe
    mage_minus = pd.DataFrame({
        'start': [data['time'].iloc[0]],
        'end': [data['time'].iloc[-1]],
        'MAGE': [abs(np.mean(mage_minus_heights)) if len(mage_minus_heights) > 0 else np.nan],
        'plus_or_minus': ['MINUS'],
        'first_excursion': [not is_plus_first]
    })

    # Determine which direction has maximum MAGE value
    ((mage_plus['MAGE'].iloc[0] >= mage_minus['MAGE'].iloc[0])
                    if not pd.isna(mage_plus['MAGE'].iloc[0])
                    and not pd.isna(mage_minus['MAGE'].iloc[0])
                    else False
    )

    return pd.concat([mage_plus, mage_minus], ignore_index=True)




def calculate_mage_plus(differences, minmax, standardD):
    """
    Calculate MAGE+ (positive glycemic excursions)

    Args:
        differences: NxN matrix of pairwise differences between extrema
        minmax: Array of extrema values (peaks and nadirs)
        standardD: Standard deviation threshold

    Returns:
        tuple: (mage_plus_heights, mage_plus_tp_pairs)
    """
    N = len(minmax)
    mage_plus_heights = []
    mage_plus_tp_pairs = []
    j = prev_j = 0  # Python uses 0-based indexing

    while j < N:
        # Get differences from previous extrema to current point j
        delta = differences[prev_j:j+1, j]  # j+1 because Python slicing is exclusive

        if len(delta) == 0:
            j += 1
            continue

        max_v = np.max(delta)  # Find maximum upward movement
        i = int(np.argmax(delta) + prev_j)  # Index of extrema creating maximum

        if max_v > standardD:
            # Found significant upward excursion (nadir to peak > SD)
            k = j
            while k < N:
                if minmax[k] >= minmax[j]:
                    j = k  # Continue riding the peak upward

                # Check if excursion ends (significant drop or end of data)
                if differences[j, k] < -standardD or k == N - 1:
                    max_v = minmax[j] - minmax[i]
                    # Record the excursion
                    mage_plus_heights.append(max_v)
                    mage_plus_tp_pairs.append((i, j))  # (nadir_index, peak_index)

                    prev_j = k
                    j = k
                    break
                k += 1
        else:
            j += 1

    return mage_plus_heights, mage_plus_tp_pairs

def calculate_mage_minus(differences, minmax, standardD):
    """
    Calculate MAGE- (negative glycemic excursions)

    Args:
        differences: NxN matrix of pairwise differences between extrema
        minmax: Array of extrema values (peaks and nadirs)
        standardD: Standard deviation threshold

    Returns:
        tuple: (mage_minus_heights, mage_minus_tp_pairs)
    """
    N = len(minmax)
    mage_minus_heights = []
    mage_minus_tp_pairs = []
    j = prev_j = 0  # Python uses 0-based indexing

    while j < N:
        # Get differences from previous extrema to current point j
        delta = differences[prev_j:j+1, j]  # j+1 because Python slicing is exclusive

        if len(delta) == 0:
            j += 1
            continue

        min_v = np.min(delta)  # Find maximum downward movement (most negative)
        i = np.argmin(delta) + prev_j  # Index of extrema creating minimum

        if min_v < -standardD:  # Found significant downward excursion
            k = j
            while k < N:
                if minmax[k] <= minmax[j]:
                    j = k  # Continue riding the nadir downward

                # Check if excursion ends (significant rise or end of data)
                if differences[j, k] > standardD or k == N - 1:
                    min_v = minmax[j] - minmax[i]  # Calculate final excursion magnitude
                    # Record the excursion (note: min_v will be negative)
                    mage_minus_heights.append(min_v)
                    mage_minus_tp_pairs.append((i, j, k))  # (peak_index, nadir_index, end_index)

                    prev_j = j
                    j = k
                    break
                k += 1
        else:
            j += 1

    return mage_minus_heights, mage_minus_tp_pairs

def segment_time_series(data, max_gap_minutes):
    """
    Split glucose time series into segments based on large gaps
    Simpler approach using time differences
    """
    # Calculate time differences

    # Calculate time differences between consecutive non-NA glucose readings
    data['time_diff'] = np.nan
    valid_indices = data['gl'].notna()
    if valid_indices.any():
        # Get timestamps of valid readings
        valid_times = data.loc[valid_indices, 'time']
        # Calculate differences between consecutive valid readings
        time_diffs = valid_times.diff().dt.total_seconds() / 60  # Convert to minutes
        # Assign differences back to original dataframe at valid indices
        data.loc[valid_indices, 'time_diff'] = time_diffs

    # Identify where gaps exceed threshold
    large_gaps = data['time_diff'] > max_gap_minutes

    # Create segment labels by cumulatively summing large gaps
    # This creates a new segment ID each time we encounter a large gap
    data['segment_id'] = large_gaps.cumsum()

    # Group by segment and return list of DataFrames
    segments = []
    for _segment_id, group in data.groupby('segment_id'):
        # Drop the temporary columns we added
        group = group.drop(['time_diff', 'segment_id'], axis=1)
        # Drop rows with NA glucose values at the end of the segment
        while len(group) > 0 and pd.isna(group['gl'].iloc[-1]):
            group = group.iloc[:-1]
        segments.append(group.reset_index(drop=True))

    return segments
    # Identify where gaps exceed threshold
