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


def execute_iglu_method(iglu_method_name: str, df: pd.DataFrame, **kwargs):

    # Get the function object
    method = getattr(iglu, iglu_method_name)

    results = method(df, **kwargs)

    return results

def run_scenario(scenarios: list[str], input_file_name: str):

    df = pd.read_csv(input_file_name,index_col=0)

    run_dict = {"input_file_name": input_file_name, "scenarios": []}

    for scenario in scenarios:
        scenario_dict = {"method": scenario[0], "kwargs": scenario[1] if len(scenario) > 1 else {}}
        results=execute_iglu_method(scenario[0], df, **scenario[1] if len(scenario) > 1 else {} ) # overrides defaults
        scenario_dict["results"] = results.to_dict()
        run_dict["scenarios"].append(scenario_dict)

    return run_dict


def main():
    scenarios = [
        ["adrr"],
        ["mean_glu"],
        ["mage"],
        ["mage", {"short_ma" :3, "long_ma" : 35}],
    ]

    input_files = [
        #"tests/data/example_data_1_subject.csv",
        "tests/data/example_data_5_subject.csv",
    ]

    runs = []
    for input_file in input_files:
        run_dict = run_scenario(scenarios, input_file)
        runs.append(run_dict)

    # save to json file
    with open('expected_results.json', 'w') as f:
        json.dump(runs, f , indent=4)

if __name__ == "__main__":
    main()
