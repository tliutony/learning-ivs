import os
import ast
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as plc
from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from src.utils import Config
from src.data import TabularDataModule
from src import model as modelzoo

torch.multiprocessing.set_sharing_strategy('file_system')


def setup():
    """
    Setup command line arguments and load config file
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='[0]', type=str)
    parser.add_argument('--work_dir', type=str, default='')
    args = parser.parse_args()


    cfg = Config.fromfile(args.cfg)
    # fix random seed (seed in config has higher priority )
    seed = cfg.seed if cfg.get('seed', None) is not None else args.seed
    # command line arguments have higher priority
    cfg.seed = seed
    args.accelerator = 'auto'
    args.devices = ast.literal_eval(args.gpu_ids)
    if args.work_dir:
        cfg.work_dir = args.work_dir

    # reproducibility
    pl.seed_everything(seed)
    return args, cfg


def train():
    """
    Main training function
    """
    args, cfg = setup()

    # model
    model_name = cfg.model.pop('name', None)
    if model_name is not None:
        model = getattr(modelzoo, model_name)(**cfg.model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented yet. Please check the model name.")

    # data
    # prefer huggingface repo
    if 'hf_dataset' in cfg:
        print(f"Downloading dataset from HF: {cfg.hf_dataset}...")
        data_path = snapshot_download(repo_id=cfg.hf_dataset, repo_type="dataset")
        if model_name == 'TransformerEncoder': # if using transformer with hf dataset, set up online transformation
            assert 'transformer_transform' in cfg, "online transformation with transformers requires transformer_transform to be specified in cfg, must be one of \{True\}"
            assert 'window_size' in cfg, "online transformation with transformers requires window_size to be specified in train cfg"
            # update later with other transforms if needed
            if getattr(cfg, 'transformer_transform', False): # apply data transforms to hf data using TransformerDataGenerator to make data transformer-ready
                # define custom data_cfg for 'online transformation' (loading data and transforming with TransformerDataGenerator)
                cfg.data_cfg = Config(dict(
                    generation = dict(generator="TransformerDataGenerator",
                                    data_path=data_path,
                                    window_size=cfg.window_size,
                                    stage='train'),
                    # data splitting 
                    n_datasets = 10000,
                    n_train = 0.16, # 0.8,  # reduced n_train for more tractable training. # proportion of data to use for training
                    n_val = 0.1,  # proportion of data to use for validation
                    n_test = 0.74 # 0.1  # proportion of data to use for testing
                                    ))
                data_path = None # make sure TabDataMod (init below) doesn't explicitly pull data from path as final dataset
                # TransfDataGen does remaining work pulling data from specified path and manipulating it, within TabDataMod using cfg.data_cfg
        # INCLUDE OTHER CASES USING ONLINE TRANSFORMATION HERE FOR FUTURE REFERENCE
    # otherwise use local data_dir
    else:
        data_path = cfg.data_dir

    data_module = TabularDataModule(data_path, cfg.train_batch_size, cfg.val_batch_size, cfg.test_batch_size,
                                    cfg.data_cfg, use_sequence=cfg.use_sequence, sequence_length=cfg.sequence_length, use_huggingface=cfg.use_huggingface, lazy_loading=cfg.lazy_loading)
    # optimization
    args.max_epochs = cfg.max_epochs

    # log
    # save config file to log directory
    cfg_name = args.cfg.split('/')[-1]
    if os.path.exists(os.path.join(cfg.work_dir, cfg.exp_name)):
        cfg.dump(os.path.join(cfg.work_dir, cfg.exp_name, cfg_name))
    else:
        os.makedirs(os.path.join(cfg.work_dir, cfg.exp_name))
        cfg.dump(os.path.join(cfg.work_dir, cfg.exp_name, cfg_name))

    # callbacks
    callbacks = [plc.EarlyStopping(**cfg.early_stopping)]  # plc.RichProgressBar()
    # used to control early stopping
    # used to save the best model
    dirpath = os.path.join(cfg.work_dir, cfg.exp_name, 'ckpts')
    os.makedirs(dirpath, exist_ok=True)
    filename = f'exp_name={cfg.exp_name}-' + f'{{{cfg.early_stopping.monitor}:.4f}}'
    callbacks.append(plc.ModelCheckpoint(dirpath=dirpath, filename=filename, **cfg.checkpoint))
    args.callbacks = callbacks

    # logger
    save_dir = os.path.join(cfg.work_dir, cfg.exp_name, 'log')
    os.makedirs(save_dir, exist_ok=True)
    if cfg.get('logging', True):
        args.logger = [WandbLogger(name=cfg.exp_name, project=cfg.project_name, save_dir=save_dir)]

    # load trainer
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()


