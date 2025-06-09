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
from datetime import datetime
from importlib.metadata import version

import iglu_py as iglu
import numpy as np
import pandas as pd
import rpy2.rlike.container
import rpy2.robjects as ro
from iglu_py import bridge
from tzlocal import get_localzone


####
# Need to fix the CGMS2DayByDay function to handle R NamedList properly.
# also fixed index of the dataframe to be continuous
####
@bridge.df_conversion
def my_CGMS2DayByDay(data: pd.DataFrame, return_df=False, **kwargs):
    """Linearly interpolates the glucose data & splits it into 24-hour periods

    If `return_df = False`, returns a dictionary with 3 keys:
        - dt0: a float that is the (1 / sampling_frequency). For exampke, if sampling_frequency = 1 / 5 min, dt0 = 5 min
        - gd2d: N x (1440 min / dt0) numpy array. Each row represents a day w/ the linearly interpolated glucose values at [0 (midnight), dt0, 2*dt0, ..., 1440 min].
        - actual_dates: list of length N, with each date as a PD Date TimeStamp at at midnight (00:00:00)

    If `return_df = True`, then formats the above data into a pd dataframe with 3 columns: (id, time, gl)
        - NOTE: there may be some rows with gl value of "nan"
    """
    if len(set(data["id"].tolist())) != 1:
        raise ValueError(
            "Multiple subjects detected. This function only supports linear interpolation on 1 subject at a time. Please filter the input dataframe to only have 1 subject's data"
        )

    r_named_list = bridge.iglu_r.CGMS2DayByDay(data, **kwargs)

    result = {
        name: ro.conversion.rpy2py(r_named_list[i])
        for i, name in enumerate(r_named_list.names())
    }

    result["actual_dates"] = [
        pd.to_datetime(d, unit="D", origin="1970-01-01") for d in result["actual_dates"]
    ]
    result["dt0"] = result["dt0"][0]

    if return_df:
        df = pd.DataFrame({"id": [], "time": [], "gl": []})
        for day in range(result["gd2d"].shape[0]):
            gl = result["gd2d"][day, :].tolist()
            n = len(gl)
            time = [
                pd.Timedelta(i * result["dt0"], unit="m") + result["actual_dates"][day]
                for i in range(n)
            ]

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {"id": n * [data["id"].iat[0]], "time": time, "gl": gl}
                    ),
                ],
                ignore_index=True,
            )

        return df
    return result

@bridge.df_conversion
def my_episode_calculation(data: pd.DataFrame, **kwargs):
    results = bridge.iglu_r.episode_calculation(data, **kwargs)

    if isinstance(results, pd.DataFrame):
        return results

    # here we have to handle NamedList
    assert isinstance(results,rpy2.rlike.container.NamedList)

    results_dict = {}

    for i, name in enumerate(results.names()):
        new_df = ro.conversion.rpy2py(results[i])
        new_df = new_df.replace({np.nan: None})  # To keep JSON happy
        new_df_dict = new_df.to_dict()
        if name in results_dict:
            results_dict[name].update(new_df_dict)
        else:
            results_dict[name] = new_df_dict
    return results_dict

def convert_timestamps_to_str(obj):
    """Convert pandas Timestamp objects to strings in a dictionary or list."""
    if isinstance(obj, pd.Timestamp):
        #return obj.strftime("%Y-%m-%d %H:%M:%S")
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_timestamps_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_str(item) for item in obj]
    return obj


def convert_nan_to_none(obj):
    """Convert NaN values to None in a dictionary or list."""
    if isinstance(obj, dict):
        return {key: convert_nan_to_none(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, (float, np.float32, np.float64, np.float16, np.longdouble)):
        if np.isnan(obj):
            return None
        else:
            return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        if np.isnan(obj):
            return None
        else:
            return int(obj)  # To keep JSON happy
    return obj


def execute_iglu_method(iglu_method_name: str, df: pd.DataFrame, **kwargs):
    # Get the function object
    method = getattr(iglu, iglu_method_name)

    results = method(df, **kwargs)

    return results


def run_scenario(scenarios: list[str], input_file_name: str):
    df = pd.read_csv(input_file_name, index_col=0)
    df['gl'] = df['gl'].astype(float)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    run_results = []

    for scenario in scenarios:
        scenario_dict = {
            "method": scenario[0],
            "input_file_name": input_file_name,
            "kwargs": scenario[1] if len(scenario) > 1 else {},
        }
        if scenario[0] == "CGMS2DayByDay":
            results = my_CGMS2DayByDay(df, **scenario[1] if len(scenario) > 1 else {})
        elif scenario[0] == "episode_calculation":
            results =my_episode_calculation(df, **scenario[1] if len(scenario) > 1 else {})
        else:
            results = execute_iglu_method(
                scenario[0], df, **scenario[1] if len(scenario) > 1 else {}
            )

        # Convert DataFrame to dict and handle timestamps
        if isinstance(results, pd.DataFrame):
            results = results.replace({np.nan: None})  # To keep JSON happy
            results.reset_index(drop=True, inplace=True)
            results_dict = results.to_dict()
        else:
            assert isinstance(results, dict), "Results must be a dictionary"
            results_dict = results

        scenario_dict["results"] = results_dict
        run_results.append(scenario_dict)

    return run_results


#  this is a hack to handle NaN values in numpy arrays
class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen = set()  # Track objects we've seen

    def default(self, obj):
        # Get object id to detect circular references
        obj_id = id(obj)
        if obj_id in self._seen:
            return f"<circular reference to {type(obj).__name__}>"
        self._seen.add(obj_id)

        try:

            if isinstance(
                obj,
                (float, np.number, np.float32, np.float64, np.float16, np.longdouble),
            ):
                # Handle all NaN variations
                if pd.isna(obj) or np.isnan(obj):
                    return None
                else:
                    return float(obj)

            if isinstance(obj, (int, np.int32, np.int64)):
                if pd.isna(obj) or np.isnan(obj):
                    return None
                else:
                    return int(obj)  # To keep JSON happy

            # Handle pandas Timestamp
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()

            # Handle numpy datetime64
            if isinstance(obj, np.datetime64):
                return pd.Timestamp(obj).isoformat()

            # Handle datetime
            if isinstance(obj, datetime):
                return obj.isoformat()

            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return [self.default(item) for item in obj]

            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                return [self.default(item) for item in obj]

            # Handle dictionaries
            if isinstance(obj, dict):
                return {key: self.default(value) for key, value in obj.items()}

            # If we get here, try to convert to a basic Python type
            # if hasattr(obj, 'tolist'):
            #     return obj.tolist()
            # if hasattr(obj, 'to_dict'):
            #     return obj.to_dict()

            return str(obj)  # Fallback to string representation

        finally:
            self._seen.remove(obj_id)  # Clean up after processing


def main():
    multi_subject_scenarios = [
        ["above_percent"],
        ["above_percent", {"targets_above": [50, 100, 250]}],
        ["active_percent"],
        ["active_percent", {"range_type": "manual"}],
        ["adrr"],
        ["auc"],
        ["below_percent"],
        ["below_percent", {"targets_below": [30, 100]}],
        ["cogi"],
        ["cogi", {"targets": [80, 180]}],
        ["conga"],
        ["conga", {"n": 96}],
        ["cv_glu"],
        ["cv_measures"],
        ["ea1c"],
        ["episode_calculation"],
        ["episode_calculation", {"return_data": True}],
        ["gmi"],
        ["grade"],
        ["grade_eugly"],
        ["grade_hyper"],
        ["grade_hypo"],
        ["gri"],
        ["gvp"],
        ["hbgi"],
        ["hyper_index"],
        ["hypo_index"],
        ["igc"],
        ["in_range_percent"],
        ["iqr_glu"],
        ["j_index"],
        ["lbgi"],
        ["m_value"],
        ["mad_glu"],
        ["mag"],
        ["mage"],
        ["mage",{"version":"naive"}],
        ["mean_glu"],
        ["median_glu"],
        ["modd"],
        ["pgs"],
        ["quantile_glu"],
        ["range_glu"],
        ["roc"],
        ["sd_glu"],
        ["sd_measures"],
        ["sd_roc"],
        ["process_data", {"id": "id", "timestamp": "time", "glu": "gl"}],
        ["summary_glu"]
    ]

    multi_subject_input_files = [
        "tests/data/example_data_1_subject_with_days.csv",
        "tests/data/example_data_5_subject.csv",
    ]

    single_subject_scenarios = [
        ["CGMS2DayByDay"],
        ["CGMS2DayByDay", {"dt0": 10}],
        ["CGMS2DayByDay", {"dt0": 5, "inter_gap": 15}],
    ]

    single_subject_input_files = [
        "tests/data/example_data_Subject_1.csv",
        "tests/data/example_data_Subject_2.csv",
        "tests/data/example_data_Subject_3.csv",
        "tests/data/example_data_Subject_4.csv",
        "tests/data/example_data_Subject_5.csv",
    ]

    # record config
    config = {
        "local_tz": get_localzone().key,
        "iglu-py": version("iglu-py"),
        "iglu": str(ro.r('packageVersion("iglu")')),
        "R": str(ro.r("R.version.string")[0]),
    }

    runs = []
    for input_file in multi_subject_input_files:
        run_results = run_scenario(multi_subject_scenarios, input_file)
        runs += run_results

    for input_file in single_subject_input_files:
        run_results = run_scenario(single_subject_scenarios, input_file)
        runs += run_results

    expected_results = {"config": config, "test_runs": runs}

    expected_results = convert_timestamps_to_str(expected_results)  # To keep JSON happy
    # expected_results = convert_nan_to_none(expected_results) # To keep JSON happy
    # save to json file
    with open("tests/expected_results.json", "w") as f:
        json.dump(expected_results, f, indent=4, cls=CustomJSONEncoder)


if __name__ == "__main__":
    main()
