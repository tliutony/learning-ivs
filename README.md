# Learning-IVs

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Data](#data)

## Introduction


## Getting Started

1. Access via a docker:
   1. git clone https://github.com/tliu526/learning-ivs/tree/dev
   2. cd learning-ivs/environment/podman && podman build -t liv .
   3. podman run --name=liv_kernel --gpus all --ipc=host -dit -v /home/charon/project:/project -p 7777:33 liv
   4. Now you are in the container, you can run the code in the container
   More details about podman: [KordingLab/podman](https://github.com/KordingLab/wiki/tree/master/compute/containers])

2. Access via a conda environment:
   1. git clone https://github.com/tliu526/learning-ivs/tree/dev
   2. conda env create -n liv python=3.9
   3. cd cd learning-ivs/environment/conda
   4. conda activate liv && pip install -r requirements.txt

## Usage
### File structure
```
.
├── project_root
│   ├── README.md
│   ├── src
│   │   ├── data
│   │   ├── model
│   │   ├──  utils
│   ├── scripts
│   │   ├── train.py
│   │   ├── generate_data.py
│   │   ├──  train.sh
│   ├── notebooks
│   │   ├── EDA.ipynb
│   │   ├──  model_evaluation.ipynb
│   ├── datasets
│   │   ├── linear
│   │   │   ├── linear_norm.py  # config file
│   │   ├── flu
│   │   │   ├── flu_clean.dta  # Stata file from McDonald et al. 1992
│   ├── configs
│   │   ├── linear_normal
│   │   │   ├── base.py # some base setting across didfferent baselines
│   │   │   ├── cnn_linear_normal_test.py # cnn working on the linear_normal generated data
│   │   │   ├── gcn_linear_normal_train.py # gcn working on the linear_normal generated data
│   │   │   ├── mlp_linear_norm.py # mlp working on the linear_normal generated data
│   ├── environment
```

### Training
```
python scripts/train.py --cfg configs/linear_normal/mlp_linear_norm.py --gpu_ids [0]
``` 

## Data
Generation logic:
1. src/data/iv_data_generation.py implements different kinds of data generator:
    1. LinearNormalDataGenerator (Linear equation with additive normal noise)
2. datasets/linear has configs used to control the specific parameters in the data generation
    1. linear_norm.py (n_sample_range, iv_strength_range, treat_effect_range ...)
3. Two ways of generation
   1. Online generation (generate data on the fly and feed into a data loader):
        For deep learning, most likely it will be online generation, which is calling the 
   src/data/tabular_datamodule.py on top of the iv_data_generator.py to generate data before start training
   2. Offline generation (generate data and save to csv files):
        For other usage, also allow generating data then save to csv files, which is calling the src/scripts/generate_data.py

```
# if not specified --work_dir, then it will be saved to the working directory in the config file
python src/scripts/generate_data.py --cfg datasets/linear/linear_norm.py --work_dir /tmp/linear_norm
```

