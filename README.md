# Learning-IVs

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Data](#data)

## Introduction

Code repo for "Learning to efficiently use instrumental variables". 

TLDR: we aim to benchmark modern machine learning methods on their ability to predict treatment effects from instrumental variable studies, compared to traditional methods like two-stage least-squares (TSLS).

For more details, see [this document](https://docs.google.com/document/d/1EODX4RHlNU0oUI7sM6cc-GS7Ku6-hx0oR8qe2OD8hYk/edit?usp=sharing). 

## Getting Started

### Access via conda
1. `git clone https://github.com/tliu526/learning-ivs/`
2. `conda create -n liv python=3.9`
3. `conda activate liv && pip install -r environment/conda/requirements.txt`

<!-- ## Usage -->

<!-- ### File structure
```
├── learning-ivs
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
``` -->

## Training
Run the following line to start a training run. This line is also available in `./scripts/train.sh`, which can be run instead.
```
python scripts/train.py --cfg configs/linear/train/transformer_linear_lennon7.py --gpu_ids [0]
``` 
Replace the cfg argument with a path to any other training config file. The training config file specifies all relevant hyperparamters to the run. See `./configs/linear/train` for examples of such training configs. **We suggest using one of the configs provided in the project as a starting point.**

See below for more information about config parameters.

## Data

Training data can either be generated online, loaded offline (locally), or downloaded online from a HuggingFace dataset repository. 
- For **online data generation**, we use a separate config file containing data generating parameters. In the train config, set parameter **data_cfg** to the path to the data config. See below for details about data config parameters.
- To **load data offline**, first generate data by running `./scripts/generate_data.py`, which simply generates data online, but saves this to some directory in the project. Then, in the training config, set parameter **data_dir** to this path.
- To **download data online from a HuggingFace dataset repo**, set the **hf_dataset** parameter to the path to the HF dataset. The data in our HF repo is not formatted for transformers, so if using transformers, set the **transformer_transform** parameter to True, and **window_size** to specify how many data points to concatenate as a single sequence input to the transformer.
### Data Config Parameters
To generate data, we implement DataGenerator classes. The **generation** parameter in the data config should specify the DataGenerator class to use, as well as the parameters for that generator. Below are brief descritions of the DataGenerators we implement (see `./data` for implementations)

- **LinearNormalDataGenerator:** Generates data from a linear equation with additive normal noise. 
- **LennonIVGenerator:** Generates data from a linear equation with correlated instruments, following Lennon et al. 2022. 
- **TransformerDataGenerator** wraps LinearNormalDataGenerator and performs extra transformations to make data compatible with transformer models. 


<!-- 1. src/data/iv_data_generation.py implements different kinds of data generator:
    1. LinearNormalDataGenerator (Linear equation with additive normal noise)
2. datasets/linear has configs used to control the specific parameters in the data generation
    1. linear_norm.py (n_sample_range, iv_strength_range, treat_effect_range ...)
3. Two ways of generation
   1. Online generation (generate data on the fly and feed into a data loader):
        For deep learning, most likely it will be online generation, which is calling the 
   src/data/tabular_datamodule.py on top of the iv_data_generator.py to generate data before start training
   2. Offline generation (generate data and save to csv files):
        For other usage, also allow generating data then save to csv files, which is calling the src/scripts/generate_data.py -->

<!-- ```
# if not specified --work_dir, then it will be saved to the working directory in the config file
python src/scripts/generate_data.py --cfg datasets/linear/linear_norm.py --work_dir /tmp/linear_norm
``` -->

