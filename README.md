# complete_Proof.pdf
This pdf contains the complete proof of the paper, the additional experiments, the background of structured sampling, and the notation table.


# Datasets

The original datasets are available from:

- The LargeST Benchmark Dataset: kaggle.com/datasets/liuxu77/largest
- Dynamical System Multivariate Time Series: kaggle.com/datasets/patrickfleith/dynamical-system-multivariate-time-series-forecast
- RealWorld (HAR) 2016: uni-mannheim.de/dws/research/projects/activity-recognition/
- The USC-SIPI Human Activity Dataset: sipi.usc.edu/had/

Process the original datasets using the Python scripts in `Data/` before running experiments.

# Dependencies

Activate your Python environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

# Unified launcher and path settings

Run experiments through `scripts/launch_common.sh`. The launcher requires Bash 4+
and GNU coreutils; use Linux or WSL. A fixed Conda environment and `screen` are not required.

All path settings are in the **USER PATH SETTINGS** section at the top of the
launcher. Variable names are case-sensitive: use `DATA_ROOT` and `RUN_ROOT`.

## Edit the two root paths

You can edit these two lines directly. Their initial values are:

```bash
DATA_ROOT="$(absolute_path "${DATA_ROOT:-$PROJECT_ROOT/datasets}")"
RUN_ROOT="$(absolute_path "${RUN_ROOT:-$PROJECT_ROOT/outputs}")"
```

For example, if your datasets are in `/srv/datasets` and experiment outputs should
go to `/srv/experiments`, replace the default paths while keeping the surrounding syntax:

```bash
# Set the default directory containing your preprocessed datasets.
DATA_ROOT="$(absolute_path "${DATA_ROOT:-/srv/datasets}")"
# Set the default directory for results, logs, and checkpoints.
RUN_ROOT="$(absolute_path "${RUN_ROOT:-/srv/experiments}")"
```

These are example paths; replace them with real directories on your server.
Prepare the dataset files beforehand. Output directories are created as needed
when experiments run. Exported environment variables take precedence over these defaults.

If datasets are stored in separate locations, edit `HAR_ROOT`, `USC_ROOT`,
`TRAFFIC_ROOT`, `DYNAMIC_ROOT`, or `SMD_ROOT` in the same section. For example:

```bash
# Override the HAR directory if it is outside DATA_ROOT.
HAR_ROOT="$(absolute_path "${HAR_ROOT:-/storage/HAR_dataset}")"
```

## What absolute_path does

`absolute_path` is a helper function already defined in the launcher, not a setting
that you need to supply:

```bash
absolute_path() { realpath -m -- "$1"; }
```

It converts a path to an absolute path and normalizes `.` and `..` components.
For example, when you launch from `/srv/project`, `./datasets` becomes
`/srv/project/datasets`. An absolute path such as `/srv/datasets` still refers to
that directory. The function does not download, move, or create files, and it does
not validate dataset contents. The `-m` option allows an output directory to be
missing; the launcher checks required data files separately before an actual run.

The outer `$(...)` captures the path returned by the function.
`${DATA_ROOT:-...}` uses `DATA_ROOT` if it is already set and nonempty; otherwise,
it uses the default after `:-`. Usually you only need to edit the default paths.
You do not need to change this helper or configure `PROJECT_ROOT`.

## Set paths from the terminal

You can also set paths through environment variables in Bash without editing the file:

```bash
# Use your own dataset and output directories.
export DATA_ROOT="/path/to/datasets"
export RUN_ROOT="/path/to/experiment_outputs"

# Optional: use a specific Python environment.
export PYTHON="/path/to/environment/bin/python"

# Check resolved paths and preview commands before starting training.
bash scripts/launch_common.sh classification --show-paths
bash scripts/launch_common.sh classification --dry-run
bash scripts/launch_common.sh classification
```

To override paths for a single invocation:

```bash
HAR_ROOT="/path/to/HAR" RUN_ROOT="/path/to/results" \
  DATASETS=HAR bash scripts/launch_common.sh classification --dry-run
```

By default, datasets are read from `datasets/` and outputs are written to `outputs/`
inside the project. These directories are separate from `Data/`, which contains
preprocessing code. Paths may contain spaces. Relative path overrides are resolved
from the caller's working directory; default paths are anchored to the project directory.
The launcher does not download datasets or install dependencies.

| Variable | Default under `DATA_ROOT` | Required preprocessed files |
| --- | --- | --- |
| `HAR_ROOT` | `HAR_dataset/` | `training_data.csv`, `validation_data.csv`, `testing_data.csv` |
| `USC_ROOT` | `USC_dataset/` | `training_data.csv`, `validation_data.csv`, `testing_data.csv` |
| `TRAFFIC_ROOT` | `traffic/` | `traffic.csv`; date column plus 371 features |
| `DYNAMIC_ROOT` | `DynamicMTS/` | `dynamic_data.csv`; date column plus 17 features |
| `SMD_ROOT` | `SMD/machine-2-3/` | `train.csv`, `test.csv`, `test_label.csv` |

Set `TRAFFIC_DATA_PATH` or `DYNAMIC_DATA_PATH` to change forecasting filenames;
absolute CSV paths are also supported. `CACHE_ROOT` and `RUNTIME_TMP_ROOT` control
cache and temporary files, defaulting to `RUN_ROOT/cache` and `RUN_ROOT/tmp`.
Changing path settings does not move, copy, or modify existing data.

# Run experiments

Replace `TASK` with a task from the table below:

```bash
bash scripts/launch_common.sh TASK --dry-run
bash scripts/launch_common.sh TASK
```

| TASK | Experiments |
| --- | --- |
| `classification` | HAR / USC classification, DLinear and PatchTST |
| `forecasting` | LargeST / DynamicMTS forecasting, DLinear and PatchTST |
| `anomaly_detection` | SMD privacy accounting, training, and summary |
| `concept_drift` | LargeST / USC PatchTST training and gradient covariance analysis |
| `structured_classification` | HAR / USC structured sampling with replacement |
| `structured_forecasting` | LargeST / DynamicMTS structured sampling with replacement |
| `spas_prepare` | Generate perturbed HAR / USC training data |
| `spas_classification` | Train classifiers on SPAS perturbed data |
| `all` | Run classification, forecasting, anomaly detection, and concept drift sequentially |

Every task supports `--help`, `--show-paths`, and `--dry-run`. A dry run does not
read datasets, execute Python, or create output files. `all` does not run structured
sampling or SPAS. Run tasks separately when applying task-specific filters such as `DATASETS`.

Classification, forecasting, anomaly detection, and concept drift support `--resume`:

```bash
bash scripts/launch_common.sh forecasting --resume
```

Classification and forecasting run each seed separately. Resume skips completed
results and checks existing command signatures. Anomaly detection retains its data,
configuration, and code identity checks. Concept drift reuses complete training
checkpoints and reruns the analysis. Structured sampling and SPAS do not support
`--resume`; they reject existing output directories, so select a new output location.

Shell scripts in the original task directories are compatibility entry points.
All configuration and launch logic live in the unified launcher. For example,
`bash classification/launch.sh --dry-run` is equivalent to the corresponding unified command.
`SPAS/HAR_DLinear_single.sh` and `SPAS/USC_DLinear_single.sh` default to HAR and USC, respectively.

# Outputs and experiment options

| Task | Default output under `RUN_ROOT` |
| --- | --- |
| classification | `classification/{results,logs,checkpoints}/` |
| forecasting | `forecasting/{traffic,DynamicMTS}/{results,logs}/`, `forecasting/checkpoints/` |
| anomaly_detection | `anomaly_detection/confirmation/L250_s130/`; development uses `screen/L250_s130/` |
| concept_drift | `concept_drift/results/`, `concept_drift/checkpoints/` |
| structured_classification | `structured_sampling/classification/<HAR or USC>/` |
| structured_forecasting | `structured_sampling/forecasting/<traffic or DynamicMTS>/` |
| spas_prepare | `SPAS/noised_data/<HAR or USC>/` |
| spas_classification | `SPAS/classification/<dataset>/<model>/epsilon_<value>/run_<index>/` |

Task output roots are also configured in USER PATH SETTINGS. `OUTPUT_ROOT` can
override the current task's output root. Classification additionally supports
`RESULT_ROOT`, `LOG_ROOT`, and `CHECKPOINT_ROOT`. Forecasting supports
`TRAFFIC_OUTPUT_ROOT`, `DYNAMIC_OUTPUT_ROOT`, and `CHECKPOINT_ROOT`.
For concept drift, use `RESULT_ROOT` and `CHECKPOINT_ROOT` to override analysis
results and training checkpoints separately. Anomaly detection supports `DATA_DIR`,
`STUDY_ROOT`, `OUTPUT_DIR`, and `SPLIT`. `SPAS_DATA_ROOT` controls where SPAS writes
perturbed training data, independently of its classification output directory.

Classification and forecasting retain their training parameters. Select datasets,
models, methods, and seeds with environment variables:

```bash
DATASETS=USC MODELS=PatchTST METHODS=poisson EPSILONS=6 SEEDS="42 43" \
  bash scripts/launch_common.sh classification --dry-run
```

The default seeds are 42-71. Override them with `NUM_RUNS` / `SEED_START`, or `SEEDS`.
`EPSILON_LIST` takes precedence over `EPSILONS`. The default privacy budgets are
`0.1 1 1.5 ... 6`; standard LargeST/PatchTST forecasting defaults to 0.1 only.
`baseline` means ordinary Poisson sampling, `poisson` means Poisson spaced sampling,
and `stratified` means fixed-size spaced sampling. These differ from structured
sampling with replacement.

| Task / model | Learning rate | Epochs |
| --- | --- | --- |
| HAR / DLinear | 0.02 | 100 |
| USC / DLinear | 0.1 | 100 |
| HAR, USC / PatchTST | 0.01 | 8 |
| LargeST / DLinear | 0.02 | 100 |
| LargeST / PatchTST | 0.005 | 100 |
| DynamicMTS / DLinear | 0.005 | 100 |
| DynamicMTS / PatchTST | 0.0015 | 100 |

The classification window length is 200; forecasting input and prediction lengths
are 80 and 20. See `--help` and the unified launcher for additional settings.
All tasks run sequentially in the foreground. Structured sampling can use `GPU_LIST`
to assign successive jobs to the listed GPUs; it does not start the old background
Conda/screen scheduling. Use `DEVICE=cpu` for CPU execution or `DEVICE=cuda:N`
to select a GPU.

For SPAS, generate perturbed data before running classification:

```bash
DATASETS=USC EPSILONS=1 NUM_RUNS=1 bash scripts/launch_common.sh spas_prepare
DATASETS=USC EPSILON=1 TRAIN_INDEX=0 bash scripts/launch_common.sh spas_classification
```

Python entry points can still run independently, but data paths must be supplied
explicitly. The unified launcher avoids maintaining paths in multiple files.
Full training has not been run as part of this repository cleanup.

## Sampling stride

All three classification runners and all three forecasting runners accept
`--sampling_stride` (a positive integer). It controls the distance in raw
timestamps between candidate training-window starts: `0, stride, 2*stride, ...`.
It is independent of PatchTST's `--stride`, which controls patches inside a window.

- Classification defaults to `seq_len`.
- Forecasting defaults to `seq_len + pred_len` (input plus prediction window).
- Smaller values allow overlapping training windows; larger values leave gaps.
- Validation/test window spacing is unchanged. Sampling rates, lambda selection,
  and privacy accounting use the configured training stride and raw split size.

For example, add `--sampling_stride 50` to a Python training command. For the
Bash launchers, set `SAMPLING_STRIDE=50`, e.g.
`SAMPLING_STRIDE=50 bash classification/launch.sh` with the dataset roots set
as above. Without the environment override, the unified launcher uses the
reference stride (classification: 100; forecasting: 50). The Python entry points
retain the length-based defaults described above.

Checkpoint setting names and launcher result filenames include `_ss<stride>`
to distinguish runs with different sampling strides. The forecasting sampler
also includes the last valid candidate window, which the previous window-count
calculation could omit.
