"""
This script is used to build the expected results for the tests.

It uses the data from the data/ directory to build the expected results with iglu-py package

Results to be stored in expected_results.json file - it will be used to compare with the actual results of the tests.

It uses the following libraries:
- pandas
- numpy
- iglu_py
"""

import json
import pandas as pd
import numpy as np
import iglu_py as iglu
from datetime import datetime


def convert_timestamps_to_str(obj):
    """Convert pandas Timestamp objects to strings in a dictionary or list."""
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, dict):
        return {key: convert_timestamps_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_str(item) for item in obj]
    return obj


def execute_iglu_method(iglu_method_name: str, df: pd.DataFrame, **kwargs):
    # Get the function object
    method = getattr(iglu, iglu_method_name)

    results = method(df, **kwargs)

    return results


def run_scenario(scenarios: list[str], input_file_name: str):
    df = pd.read_csv(input_file_name, index_col=0)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    run_results = []

    for scenario in scenarios:
        scenario_dict = {
            "method": scenario[0], 
            "input_file_name": input_file_name, 
            "kwargs": scenario[1] if len(scenario) > 1 else {}
        }
        results = execute_iglu_method(scenario[0], df, **scenario[1] if len(scenario) > 1 else {})
        
        # Convert DataFrame to dict and handle timestamps
        results_dict = results.to_dict()
        results_dict = convert_timestamps_to_str(results_dict)
        
        scenario_dict["results"] = results_dict
        run_results.append(scenario_dict)

    return run_results


def main():
    scenarios = [
        ["above_percent"],
        ["active_percent"],
        ["active_percent", {"dt0": 5, "tz": 'GMT'}],
        ["adrr"],
        ["auc"],
        ["range_glu"],
        ["iqr_glu"],
        ["conga"],
        ["mad_glu"],
        ["mag"],
        ["mage"],
        ["modd"],
        ["j_index"],
        ["igc"],
        ["cogi"],
        ["m_value"],
        ["grade"],
        ["grade_eugly"],
        ["grade_hyper"],
        ["grade_hypo"],
        ["hyper_index"],
        ["hypo_index"],
        ["ea1c"],
        ["gvp"],
        ["hbgi"],
        ["lbgi"],
        ["sd_roc"],
        ["mean_glu"],
        ["mage"],
        ["mage", {"short_ma": 3, "long_ma": 35}],
    ]

    input_files = [
        #"tests/data/example_data_1_subject.csv",
        "tests/data/example_data_5_subject.csv",
    ]

    runs = []
    for input_file in input_files:
        run_results = run_scenario(scenarios, input_file)
        runs += run_results

    # save to json file
    with open('tests/expected_results.json', 'w') as f:
        json.dump(runs, f, indent=4)


if __name__ == "__main__":
    main()
