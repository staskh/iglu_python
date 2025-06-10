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

## Unit Test Status
Unless noted, iglu-r test is considered successful if it achieves precision of 0.001

| Function | IGLU-R test compatibility | array/list/Series | TZ | Comments |
|----------|---------------------------|-------------------|----|----------|
| above_percent | ✅ | |||
| active_percent | ✅ |
| adrr | ✅ |
| auc| 🟡 (0.01 precision) | || see [auc_evaluation.ipynb](https://github.com/staskh/iglu_python/blob/main/notebooks/auc_evaluation.ipynb)|
| below_percent| ✅ |
| cogi | ✅ |
| conga | ✅ |
| cv_glu | ✅ |
| cv_measures | ✅ |
| ea1c | ✅ |
| episode_calculation |  🟡  need fix in excl| || no match in lv1_hypo_excl and lv1_hyper_excl|
| gmi | ✅ |
| grade_eugly | ✅ |
| grade_hyper | ✅ |
| grade_hypo | ✅ |
| grade | ✅ |
| gri | ✅ |
| gvp | ✅ |
| hbgi | ✅ |
| hyper_index | ✅ |
| hypo_index | ✅ |
| igc | ✅ |
| j_index | ✅ |
| lbgi | ✅ |
| mad_glu | ✅ |
| mag |  ✅ | || IMHO, Original R implementation has an error |
| mage | ✅ | || See algorithm at [MAGE](https://github.com/irinagain/iglu/blob/master/vignettes/MAGE.Rmd) |
| mean_glu | ✅ |
| median_glu | ✅ |
| modd | ✅ |
| pgs |  ✅  | || |
| quantile_glu |  ✅ |
| range_glu | ✅ |
| roc | ✅ |
| sd_glu | ✅ |
| sd_measures | ✅ |
| sd_roc |  ✅ | |||
| process_data | ✅ |
| summary_glu | ✅ |
| CGMS2DayByDay | ✅ |

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
    'time': pd.date_range(start='2023-01-01', periods=100, freq='5min')
    'gl': [120, 135, 140, 125, 110, ...],  # glucose values in mg/dL
})

# Calculate glucose metrics
mean_glucose = iglu.mean_glu(data)
cv = iglu.cv_glu(data)
time_in_range = iglu.active_percent(data, lltr=70, ultr=180)

print(f"Mean glucose: {mean_glucose}")
print(f"CV: {cv}")
print(f"Time in range (70-180 mg/dL): {time_in_range}%")
```

### Using with Time Series Data

```python
import pandas as pd
import iglu_python as iglu
from datetime import datetime, timedelta

# Create time series data
timestamps = pd.date_range(start='2023-01-01', periods=288, freq='5min')
glucose_values = [120 + 20 * np.sin(i/48) + np.random.normal(0, 5) for i in range(288)]

data = pd.DataFrame({
    'id': ['Subject1'] * 288,
    'time': timestamps,
    'gl': glucose_values
})

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
