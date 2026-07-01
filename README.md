# master-cleaned-repo

This is the repository for the code of the masters thesis "Wavelet-Domain MLPs for Efficient Synthetic Time Series Generation via Flow Matching".

# datasets

The datasets used in this thesis are not included in this repository, but can be obtained from the following sources:

- stock 1: [https://www.kaggle.com/datasets/varpit94/google-stock-data]
- EEG 2: [https://archive.ics.uci.edu/dataset/264/eeg+eye+state]
- Dataset 3: [https://github.com/zhouhaoyi/ETDataset/tree/main]

The code is organized into the following directories:

training_n_step.py is the main script for training the model. It contains the command line arguments, and does model setup.

fm_train.py contains the code for training the models and for sampling.

metrics contains the code for calculating the evaluation metrics used in the thesis.

models contains the code for the models, transforms and the loss functions.

data contains the code for loading the datasets and for preprocessing.

To add a new dataset create a new config file in the data_configs.py file and add the argument for the dataset name, if the file is of format .csv or .arff, it should work without any code changes.

# running training

The main entry point is `training_n_step.py`.

The repostiory uses `uv` for env managment. uv run should automatically install any missing dependencies.

The code is tested and setup for python 3.12 with torch 2.10 and Cuda 12.8
Example:

```bash
uv run python -u training_n_step.py \
  --window_size 128 \
  --dataset google_stock \
  --epochs 1000 \
  --norm z \
  --transform flatwavelet \
  --coeff_norm z \
  --wavelet_type db2 \
  --group speed
```

Use `--cpu` to force CPU execution.

## important flags

### Data and preprocessing

- `--dataset`: Dataset config to load. Current configs include `google_stock`, `RICO`, `etth1`, and `eeg`.
- `--window_size`: Time-series window length used by the dataset loader.
- `--norm`: Normalizes the input time series before transformation. Supported values are `minmax` and `z`.
- `--transform`: Transform applied before training. Options are `flatwavelet` and `identity`. `identity` applies no transform and trains in the time domain.
- `--wavelet_type`: Wavelet family for `flatwavelet`, for example `db2`. The default `auto` selects from `window_size`.
- `--levels`: Number of wavelet decomposition levels. The default `-1` selects automatically.
- `--target_seq_len`: Target coefficient length used by automatic level selection. The default is `64`, which means that the level decomposes til the approximation coefficients are approximately `64` in length. This is only used if `--levels` is set to `-1`.

### Coefficient normalization

- `--coeff_norm`: Normalizes transformed coefficients. Supported values are `minmax` and `z`.

### Splitting coefficients into models

Only one split mode can be selected at a time. If no split mode is selected, a model is trained for each wavelet band.

- `--no_split`: Trains one model on all coefficients.
- `--splits`: Cumulative coefficient split boundaries. If the final split is smaller than the total coefficient count, the remainder is added unless `--drop_remainders` is set.
- `--half_split`: Splits coefficients into low- and high-frequency halves.
- `--level_split`: Splits by wavelet level indexes.
- `--keep_n_groups`: With `--no_split`, keeps the first `n` wavelet groups together.
- `--drop_remainders`: Drops coefficients after the final split boundary instead of adding a final remainder split.

### Model and training

- `--model_class`: Model class to use. Options include `SimpleMLPODE`, `TransformerODE`, `MixerODE`, and `IdentityODE`. IdentityODE is a empty model that simply returns the input, used when we don't want to learn the coefficients but instead just use the prior. If not specified the model will default to `SimpleMLPODE`. Multiple model classes can be spepecified, and they will be used in order for each split, with the last one reused if there are more splits than models.
- `--epochs`: Number of epochs. Can take multiple values, one per split/model. If fewer values are given than models, the last value is reused.
- `--lr`: Learning rate.
- `--loss`: Loss function. Options are `mse`, `new_recon_loss`, and `new_combined_loss`.
- `--alpha`: Weight used by combined losses.
- `--cond`: Conditions later-stage models on earlier coefficient groups.
- `--disable_ot`: Disables optimal transport pairing during training.
- `--shaped_prior`: Uses the shaped prior wrapper for sampling and training.

### Logging and evaluation

- `--logger`: Logger backend. Options are `mlflow` and `wandb`. If omitted, an empty logger is used.
- `--group`: Optional run group name for the logger.
- `--artifact_folder`: Folder for saved artifacts and generated data.
- `--disable_eval`: Skips final evaluation. This is useful for quick CPU/debug runs because some metrics are expensive.
- `--mlflow_path` and `--mlflow_db`: Configure MLflow output.
