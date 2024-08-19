exp_name = "lennon100_fixed_tau"

# directories
result_dir = f"./results/{exp_name}/"
data_dir = f"./datasets/linear/{exp_name}/"

# TODO should add functionality for pulling from huggingface
# TODO recreate with config
models = {
    "TSLS": {},
    "LIML": {},
    "OLS": {},
    "MHML": {},
    "PoolingMLP": {
        "chkpt_path": "/home/tliu/learning-ivs/workdir/pool_mlp_lennon_bs256_lr0.004_eps100_hidden256_depth4/ckpts/exp_name=pool_mlp_lennon_bs256_lr0.004_eps100_hidden256_depth4-val_loss=0.0070.ckpt" # This path will also be saved in the results file
    }
}

# metrics to compile for each model, kwargs for each metric
metric_funcs = {
    "mse_decomp": {
        # TODO reparameterize this
        "ground_truth": 1
    }
}
