"""
Script for running benchmark suite of models against a particular dataset.

Implementation plan:
- consume the same format config as train.py
- load the dataset
- evaluate models, shoud be extensible to any non-metalearned model
"""

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to configuration file")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--work_dir', type=str, default='')
    args = parser.parse_args()

    # load config
    cfg = Config.load(args.cfg)
    seed = cfg.seed if cfg.get('seed', None) is not None else args.seed

    # load dataset
    dataset = load_dataset(cfg.data_cfg)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    # evaluate models
    for model_cfg in cfg.models:
        model = load_model(model_cfg)
        model.fit(dataset)
        model.evaluate(dataset)
        model.save_results()

