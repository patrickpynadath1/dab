import pandas as pd
import numpy as np
import yaml
import os
import pickle


# here, kwargs = the hyper-param along with the keywords
# this will return all file paths that have the desired config settings
def search_config(sampler, exp, res_dir="results", **kwargs):
    base_path = f"{res_dir}/{exp}"
    runs = [p for p in get_subdir(base_path) if sampler in p]
    matching_paths = []
    for run in runs:
        config = yaml.load(
            open(f"{base_path}/{run}/conf.yaml", "r"), Loader=yaml.FullLoader
        )
        if all([config[k] == v for k, v in kwargs.items()]):
            matching_paths.append(f"{base_path}/{run}")
    return matching_paths


# compiles the results into a tidy dataframe, containing key metrics +
# the hyper-param settings of importance
def compile_res(file_paths, *hyper_param_of_imp):
    res = {}
    for fp in file_paths:
        cur_run_dct = {}
        config = yaml.load(open(f"{fp}/conf.yaml", "r"), Loader=yaml.FullLoader)
        for hp in hyper_param_of_imp:
            hp_setting = config.get(hp, None)
            cur_run_dct[hp] = hp_setting
        try:
            metrics = pickle.load(open(f"{fp}/eval_metrics_abl.pkl", "rb"))
        except FileNotFoundError:
            continue
        for k, v in metrics.items():
            if k == "cola":
                continue
            array_v = np.array(v)
            if len(array_v.shape) <= 2:
                cur_run_dct[f"{k}_mean"] = array_v.mean()
                cur_run_dct[f"{k}_std"] = array_v.std()
            elif len(array_v.shape) == 3:
                cur_run_dct[f"{k}_mean"] = array_v[:, :, 1].mean()
                cur_run_dct[f"{k}_std"] = array_v[:, :, 1].std()

        res[fp] = cur_run_dct
    return pd.DataFrame(res).T


def get_sampler_hyperparams(sampler, conf_dir="configs"):
    dir = f"{conf_dir}/defaults/{sampler}.yaml"
    default_settings = yaml.load(open(dir, "r"), Loader=yaml.FullLoader)
    return list(default_settings.keys())


def get_subdir(parent_dir):
    list_subfolders_with_paths = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
    return list_subfolders_with_paths
