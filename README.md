IGLU_PYTHON library

# Concept
IGLU_PYTHON is a pure Python implementation of the widely-used [IGLU](https://github.com/irinagain/iglu) (Interpreting GLUcose data) package. While the original IGLU implementation (referred to as iglu-r) is highly regarded in the research community, its R-based implementation has limited its adoption outside academic settings. The existing [IGLU-PY](https://github.com/IrinaStatsLab/iglu-py) solution provides a Python-to-R bridge but still requires a complete R installation and its dependencies.

IGLU_PYTHON reimplements all IGLU metric functions natively in Python, eliminating the need for R while maintaining full compatibility with the original package. 

This project is proudly sponsored by [Pheno.AI](https://www.pheno.ai).

## IGLU-R Compatibility

A significant focus of this project has been ensuring compatibility with the original R implementation of IGLU. To achieve this:

- The test suite includes validation against the original R implementation
- Test data is generated using `tests/build_expected_values.py`, which interfaces with the R implementation through an iglu-py adaptation layer
- Expected results are stored in `tests/expected_results.json`
- Each unit test in the package compares Python implementation results against the R-generated reference values

This approach ensures that the Python implementation produces results consistent with the original R package.

### Input & Output 
The implementation maintains compatibility with the R version while following Python best practices. The metrics can be used as:

```Python
import iglu_python ias iglu

# With DataFrame input
result_df = iglu.cv_glu(data)  # data should have 'id', 'time', and 'gl' columns
# Return DataFrame with "id' and column(s) with value(s)

# With Series input (some metrics require Series with DateTimeIndex)
result_float = iglu.cv_glu(glucose_series)  # just glucose values
# returns a single float value

# Same with function that support list or ndarray
result_float = iglu.cv_glu(glucose_list)  # list of glucose values
# returns a single float value

```

## IGLU-R Compatibility Test Status
The current version of IGLU-PYTHON is test-compatible with IGLU-R v4.2.2

Unless noted, IGLU-R test compatability is considered successful if it achieves precision of 0.001

| Function | Description | IGLU-R test compatibility | list /ndarray /Series input | TZ | Comments |
|----------|-------------|-------------|-------------------|----|----------|
| above_percent | percentage of values above target thresholds| ✅ |✅ returns Dict[str,float] |||
| active_percent | percentage of time CGM was active | ✅ | ✅ only Series(DatetimeIndex) returns Dict[str,float]|
| adrr | average daily risk range | ✅ |✅ only Series(DatetimeIndex) returns float |
| auc| Area Under Curve | 🟡 (0.01 precision) |✅ only Series(DatetimeIndex) returns float  || see [auc_evaluation.ipynb](https://github.com/staskh/iglu_python/blob/main/notebooks/auc_evaluation.ipynb)|
| below_percent| percentage of values below target thresholds| ✅ | ✅ returns Dict[str,float]|
| cogi |Coefficient of Glucose Irregularity | ✅ | ✅ returns float
| conga | Continuous Overall Net Glycemic Action |✅ | ✅ only Series(DatetimeIndex) returns float
| cv_glu | Coefficient of Variation | ✅|  ✅ returns float |
| cv_measures |Coefficient of Variation subtypes (CVmean and CVsd) |✅  |✅ only Series(DatetimeIndex) returns Dict[str,float]| | 
| ea1c |estimated A1C (eA1C) values| ✅ | ✅ returns float |
| episode_calculation | Hypo/Hyperglycemic episodes with summary statistics|  ✅|  🟡 always returns DataFrame(s)|| |
| gmi | Glucose Management Indicator | ✅ | ✅ returns float |
| grade_eugly |percentage of GRADE score attributable to target range| ✅ | ✅ returns float 
| grade_hyper |percentage of GRADE score attributable to hyperglycemia| ✅ |✅ returns float 
| grade_hypo |percentage of GRADE score attributable to hypoglycemia| ✅ |✅ returns float 
| grade |mean GRADE score| ✅ | ✅ returns float |
| gri |Glycemia Risk Index | ✅ | ✅ returns float |
| gvp |Glucose Variability Percentage| ✅ | ✅ only Series(DatetimeIndex) returns float
| hbgi |High Blood Glucose Index| ✅ | ✅ returns float |
| hyper_index |Hyperglycemia Index| ✅ |✅ returns float |
| hypo_index |Hypoglycemia Index| ✅ |✅ returns float |
| igc |Index of Glycemic Control| ✅ |✅ returns float |
| in_range_percent |percentage of values within target ranges| ✅ | ✅ returns Dict[str,float]|
| iqr_glu |glucose level interquartile range|✅ |✅ returns float |
| j_index |J-Index score for glucose measurements| ✅ |✅ returns float |
| lbgi | Low Blood Glucose Index| ✅ |✅ returns float |
| m_value | M-value of Schlichtkrull et al | ✅ |✅ returns float |
| mad_glu | Median Absolute Deviation | ✅ |✅ returns float |
| mag | Mean Absolute Glucose| ✅ | ✅ only Series(DatetimeIndex) returns float ||| IMHO, Original R implementation has an error |
| mage | Mean Amplitude of Glycemic Excursions|  ✅ |✅ only Series(DatetimeIndex) returns float || See algorithm at [MAGE](https://irinagain.github.io/iglu/articles/MAGE.html) |
| mean_glu | Mean glucose value | ✅ | ✅ returns float|
| median_glu |Median glucose value| ✅ |✅ returns float |
| modd | Mean of Daily Differences| ✅ | ✅ only Series(DatetimeIndex) returns float|
| pgs | Personal Glycemic State | ✅  |✅ only Series(DatetimeIndex) returns float| || 
| quantile_glu |glucose level quantiles|  ✅ |✅ returns List[float] |
| range_glu |glucose level range| ✅ |✅ returns float|
| roc | Rate of Change| ✅ |🟡 always returns DataFrame|
| sd_glu | standard deviation of glucose values| ✅ | ✅ returns float
| sd_measures |various standard deviation subtypes| ✅ |✅ only Series(DatetimeIndex) returns Dict[str,float]|
| sd_roc | standard deviation of the rate of change| ✅ |✅ only Series(DatetimeIndex) returns float ||
| summary_glu | summary glucose level| ✅ |✅ returns Dict[str,float]|
| process_data | Data Pre-Processor | ✅ |
| CGMS2DayByDay |Interpolate glucose input| ✅ |

## Extended functionality
IGLU_PYTHON extends beyond the capabilities of the original IGLU-R package by offering enhanced functionality and improved user experience. We believe that combining these extended features with the proven reliability of IGLU-R creates a powerful synergy that benefits both the research community and wide software developers community.



| Function          | Description                              | 
|-------------------|------------------------------------------|
| load_libre()      | Load Timeseries from Libre device file (CGM reading converted into mg/dL)
| load_dexcom()     | Load Timeseries from Dexcom device file (CGM reading converted into mg/dL)

# Installation

Install IGLU_PYTHON using pip:

```bash
pip install iglu-python
```

For development installation:

```bash
git clone https://github.com/staskh/iglu_python.git
cd iglu_python
pip install -e .
```
## Examples of Use

### Basic Usage with DataFrame

```python
import pandas as pd
import iglu_python as iglu

# Load your glucose data into a DataFrame
# Expected columns: 'id' (subject identifier) and 'gl' (glucose values)
# Optional: datetime index or 'time' column
data = pd.DataFrame({
    'id': ['Subject1'] * 100,
    'time': pd.date_range(start='2023-01-01', periods=100, freq='5min'),
    'gl': [120, 135, 140, 125, 110]*20  # glucose values in mg/dL
})

# Calculate glucose metrics
mean_glucose = iglu.mean_glu(data)
cv = iglu.cv_glu(data)
active = iglu.active_percent(data)

print(f"Mean glucose: {mean_glucose['mean'][0]}")
print(f"CV: {cv['CV'][0]}")
print(f"CGM active percent: {active['active_percent'][0]}%")
```

### Using with Time Series Data

```python
import pandas as pd
import numpy as np
import iglu_python as iglu

# Create time series data
timestamps = pd.date_range(start='2023-01-01', periods=288, freq='5min')
glucose_values = [120 + 20 * np.sin(i/48) + np.random.normal(0, 5) for i in range(288)]

data = pd.Series(glucose_values, index=timestamps)

# Calculate advanced metrics
mage = iglu.mage(data)
auc = iglu.auc(data)
gmi = iglu.gmi(data)

print(f"MAGE: {mage}")
print(f"AUC: {auc}")
print(f"GMI: {gmi}")
```

### Multiple Input Formats
(Not yet fully implemented and tested)

```python
import iglu_python as iglu
import numpy as np

# Using list (assumes 5-minute intervals)
glucose_list = [120, 135, 140, 125, 110, 95, 105, 115]
mean_from_list = iglu.mean_glu(glucose_list)

# Using NumPy array
glucose_array = np.array([120, 135, 140, 125, 110, 95, 105, 115])
cv_from_array = iglu.cv_glu(glucose_array)

# Using Pandas Series with DatetimeIndex
glucose_series = pd.Series(
    data=[120, 135, 140, 125, 110, 95, 105, 115],
    index=pd.date_range(start='2023-01-01', periods=8, freq='5min')
)
sd_from_series = iglu.sd_glu(glucose_series)
```

# Notes on IGLU-R Compatibility

During our implementation and testing process, we identified several discrepancies between our Python implementation and the original R version of IGLU. While maintaining test compatibility remains a priority, we are actively working with the IGLU-R development team to investigate and resolve these issues.

## Known Implementation Differences

### Timezone Handling in check_data_columns

The function's timezone handling behavior requires clarification:
- When a specific timezone is provided, the function performs a timezone **conversion** (`tz_convert`) rather than timezone **localization** (`tz_localize`)
- This means timestamps are being transformed to the target timezone instead of being labeled with it
- The intended behavior needs to be confirmed with the original IGLU-R authors
- This difference in timezone handling may affect daily aggregation and analysis results

### CGMS2DayByDay Function
The following issues have been identified in the R implementation:

1. Timezone Handling:
   - When using `tz=UTC`, data points are shifted one day earlier than expected
   - *Status: Pending test case development to demonstrate the issue*

2. Grid Alignment:
   - Results are shifted one grid index to the left from the expected values
   - *Status: Pending test case development to demonstrate the issue*

We are maintaining test compatibility while these issues are being investigated. Updates will be provided as we receive clarification from the IGLU-R development team.

### Input Data Types
Most metric functions, in addition to a standard DataFrame, support multiple input formats for glucose readings:
- `List[float]`: Python list of glucose values
- `np.array`: NumPy array of glucose values
- `pd.Series`: Pandas Series of glucose values (with or without DatetimeIndex)

When using these sequence types (without timestamps), the functions assume a fixed 5-minute interval between measurements. For more precise analysis with variable time intervals, use the DataFrame input format with explicit timestamps or Series with DatetimeIndex .

# ToDo
- implement Series/list/array as an input for all metrics (suing Series with DatetimeIndex)
- optimize code by NOT converting arrays/Series into DataFrames
- test and implement tz='UTC' timezone assignment
- clarify functionality correctness for CGMS2DayByDay
