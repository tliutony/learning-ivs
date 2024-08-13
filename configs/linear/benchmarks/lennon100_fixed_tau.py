exp_name = "lennon100_fixed_tau"

# directories
result_dir = f"./results/{exp_name}/"
data_dir = f"./datasets/linear/{exp_name}/"

# TODO should add functionality for pulling from huggingface
models = [
    "TSLS",
    "LIML",
    "OLS",
    "MHML",
    #"Pooled MLP"
]

# metrics to compile for each model, kwargs for each metric
metric_funcs = {
    "mse_decomp": {
        # TODO reparameterize this
        "ground_truth": 1
    }
}
