"""
Script for running benchmark suite of models against a particular dataset.

Implementation plan:
- consume the same format config as train.py
- load the dataset
- evaluate models, shoud be extensible to any non-metalearned model
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser

from src.utils import Config
from src import model as modelzoo
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

    # evaluate models
    for model_name in cfg.models:
        print(f"benchmarking {model_name}...")
        if model_name is not None:
            model = getattr(modelzoo, model_name)()
            results = model.estimate_all(test_data)
            results.to_parquet(os.path.join(cfg.result_dir, f"{model_name}_results.parquet"))
        else:
            raise NotImplementedError(f"estimator {model_name} is not implemented yet. Please check the model name.")

    print("benchmark complete!")
    

