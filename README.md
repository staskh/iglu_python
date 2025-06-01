IGLU_PYTHON library

Python version of popular IGLU (Interpreting GLUcose data) package

# IGLU-R Compatibility

A significant focus of this project has been ensuring compatibility with the original R implementation of IGLU. To achieve this:

- The test suite includes validation against the original R implementation
- Test data is generated using `tests/build_expected_values.py`, which interfaces with the R implementation through an iglu-py adaptation layer
- Expected results are stored in `tests/expected_results.json`
- Each unit test in the package compares Python implementation results against the R-generated reference values

This approach ensures that the Python implementation produces results consistent with the original R package.


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
- `pd.Series`: Pandas Series of glucose values

When using these sequence types (without timestamps), the functions assume a fixed 5-minute interval between measurements. For more precise analysis with variable time intervals, use the DataFrame input format with explicit timestamps.

# ToDo
- implement Series/list/array as an input for all metrics
- test and implement tz='UTC' timezone assignment
- clarify functionality correctness for CGMS2DayByDay
- optimize code by NOT converting arrays/Series into DataFrames