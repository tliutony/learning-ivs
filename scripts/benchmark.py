"""
Script for running benchmark suite of models against a particular dataset.
"""

import datetime as dt
import json
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser

from src.utils import Config
from src import model as modelzoo
from src.utils import metrics
from src.data import TabularDataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to configuration file")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--work_dir', type=str, default='')
    args = parser.parse_args()

    # load config
    cfg = Config.fromfile(args.cfg)
    seed = cfg.seed if cfg.get('seed', None) is not None else args.seed

    # load test datasets only, as benchmark estimators don't need training
    data_module = TabularDataModule(cfg.data_dir)
    data_module.setup(stage="test")
    test_data = data_module.testset

    if args.work_dir:
        cfg.work_dir = args.work_dir

    os.makedirs(cfg.result_dir, exist_ok=True)

    # evaluate models
    print(f"Benchmarking against {cfg.exp_name}")
    # TODO parallelize this
    result_df = pd.DataFrame()
    for model_name in cfg.models:
        print(f"\tEvaluating {model_name}...")
        if model_name is not None:
            result_path = os.path.join(cfg.result_dir, f"{model_name}_results.parquet")
            try:
                results = pd.read_parquet(result_path)
            except FileNotFoundError:
                model = getattr(modelzoo, model_name)()
                results = model.estimate_all(test_data)
                results['model'] = model_name
                results.to_parquet(result_path)

            result_df = pd.concat([result_df, results])
            
        else:   
            raise NotImplementedError(f"estimator {model_name} is not implemented yet. Please check the model name.")

    print("Generating metrics...")
    result_dict = {}
    for metric_name, kwargs in cfg.metric_funcs.items():
        print(f"\tComputing {metric_name}...")
        metric_dict = {}
        try:
            metric_func = getattr(metrics, metric_name)
            for model_name in cfg.models:
                sel_df = result_df[result_df["model"] == model_name]
                metric_dict[model_name] = metric_func(result_df=sel_df, **kwargs)

        except Exception as e:
            print(f"Error occurred when attemping to compute f{metric_name}")
            raise e
        result_dict[metric_name] = metric_dict

    result_dict['timestamp'] = str(dt.datetime.now(dt.timezone.utc).astimezone())
    with open(os.path.join(cfg.result_dir, "results.json"), "w") as out_f:
        json.dump(result_dict, out_f, indent=4)
    print("Benchmark complete!")
    

