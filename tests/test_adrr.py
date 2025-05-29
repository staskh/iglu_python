import pytest
import pandas as pd
import json
import iglu_python as iglu

method_name = 'adrr'

@pytest.fixture
def test_data():
    # Load expected results
    with open('tests/expected_results.json', 'r') as f:
        expected_results = json.load(f)

    method_scenarios = [scenario for scenario in expected_results if scenario['method'] == method_name]

    for scenario in method_scenarios:
        yield scenario

def test_adrr_calculation(test_data):  
    """Test ADRR calculation against expected results"""

    input_file_name = test_data['input_file_name']
    kwargs = test_data['kwargs']

    df = pd.read_csv(input_file_name, index_col=0)

    result_df = iglu.adrr(df, **kwargs)

    assert result_df is not None

    expected_results = test_data['results']
    expected_df = pd.DataFrame(expected_results)
    expected_df = expected_df.reset_index(drop=True)

    # Compare DataFrames with precision to 0.001
    pd.testing.assert_frame_equal(
        result_df.round(3),
        expected_df.round(3),
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_names=True,
        check_exact=True,
        check_datetimelike_compat=True,
        check_categorical=True,
        check_like=True,
        check_freq=True,
        check_flags=True,
    )

