# WorldMove
Code for the paper "WorldMove: A global open data for human mobility"

## Getting Started

**step 1**: clone the repository

**step 2**: create a virtual environment with conda or virtualenv

```bash
# with conda
conda create -n move python=3.10
conda activate move
# with virtualenv
virtualenv -p python3.10 move
source move/bin/activate
```

**step 3**: install the dependencies

```bash
pip install -r requirements.txt
```

## Model Training
First, the multi-source location feature data is processed through a location feature encoder, compressing and projecting the regional characteristics into a unified embedding space that enhances model comprehension. Building upon the location embeddings, we leverage real-world human mobility data to encode physical location sequences from different cities into a unified semantic space, forming a comprehensive mobility dataset that encompasses diverse urban mobility patterns. Our diffusion model is then trained on this unified dataset.

**step 1**: train the location feature encoder
```bash
python3 loc_encoder.py --save /path/to/save_dir
```

**step 2**: train the diffusion model

Entrypoint to the diffusion model training is train.py. The script takes a configuration file as input, which specifies the dataset path, hyperparameters, and other settings. The configuration file is in the YAML format, an example is provided in the `configs` directory.

- Modify the configuration file `configs/{city}.yml` to specify the dataset path and other hyperparameters.

- Run the following command to train the diffusion model

```bash
python3 train.py --config configs/{city}.yml --save /path/to/log_dir
```

The trained model will be saved in the log directory and you can check the training process in tensorboard by running `tensorboard --logdir /path/to/log_dir`. We trained our model for 200 epochs on both ISP and MME datasets.

## Mobility Trajectory Generation
We offer a pre-trained model and a pipeline tool for generating mobility datasets for any city worldwide. The pipeline follows a straightforward process:

**step 1**: acquire population data

The population data can be obtained from WorldPop using the script:

```bash
python3 scripts/generate_pop.py --config configs/{city}.yml --save /path/to/pop_dir
```

**step 2**: generate location profiling

Location profiling data can be created by running the script:

```bash
python3 scripts/generate_profile.py --config configs/{city}.yml --save /path/to/pop_dir
```

which integrates data such as population distribution and POI attributes.


**step 3**: generate mobility data

Using the prepared location profiling and population data, mobility data can be generated with the script:

```bash
python3 scripts/generate_mobility --config configs/{city}.yml --save /path/to/pop_dir
```