#!/usr/bin/env bash
# Unified experiment launcher. Requires Bash 4+ and GNU coreutils (Linux/WSL).
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

configure_paths() {
  # ======================== USER PATH SETTINGS ========================
  # Edit the defaults below, or export variables before running this script.
  # Environment variables take precedence over these defaults.
  # Absolute paths are recommended. Relative overrides use the caller's directory.
  # Data directories contain PREPROCESSED datasets, not the preprocessing scripts.
  # Example replacements for the next two lines (use your own real directories):
  # DATA_ROOT="$(absolute_path "${DATA_ROOT:-/srv/datasets}")"
  # RUN_ROOT="$(absolute_path "${RUN_ROOT:-/srv/experiments}")"
  # absolute_path is a helper, not a setting. Keep it and edit only the default path.
  DATA_ROOT="$(absolute_path "${DATA_ROOT:-$PROJECT_ROOT/datasets}")"
  RUN_ROOT="$(absolute_path "${RUN_ROOT:-$PROJECT_ROOT/outputs}")"
  CACHE_ROOT="$(absolute_path "${CACHE_ROOT:-$RUN_ROOT/cache}")"
  RUNTIME_TMP_ROOT="$(absolute_path "${RUNTIME_TMP_ROOT:-$RUN_ROOT/tmp}")"
  # Activate your Python environment first, or set PYTHON to its executable.
  PYTHON="${PYTHON:-python}"

  # HAR and USC: training_data.csv, validation_data.csv, testing_data.csv.
  HAR_ROOT="$(absolute_path "${HAR_ROOT:-$DATA_ROOT/HAR_dataset}")"
  USC_ROOT="$(absolute_path "${USC_ROOT:-$DATA_ROOT/USC_dataset}")"
  # Forecasting CSV files: one date column followed by the feature columns.
  TRAFFIC_ROOT="$(absolute_path "${TRAFFIC_ROOT:-${TRAFFIC_DATA_ROOT:-$DATA_ROOT/traffic}}")"
  DYNAMIC_ROOT="$(absolute_path "${DYNAMIC_ROOT:-${DYNAMIC_DATA_ROOT:-$DATA_ROOT/DynamicMTS}}")"
  TRAFFIC_DATA_PATH="${TRAFFIC_DATA_PATH:-traffic.csv}"
  DYNAMIC_DATA_PATH="${DYNAMIC_DATA_PATH:-dynamic_data.csv}"
  # SMD: train.csv, test.csv, and test_label.csv for the selected machine.
  SMD_ROOT="$(absolute_path "${SMD_ROOT:-$DATA_ROOT/SMD/machine-2-3}")"

  # Each task has a separate output directory to prevent accidental mixing.
  CLASSIFICATION_OUTPUT_ROOT="$(absolute_path "${CLASSIFICATION_OUTPUT_ROOT:-$RUN_ROOT/classification}")"
  FORECASTING_OUTPUT_ROOT="$(absolute_path "${FORECASTING_OUTPUT_ROOT:-$RUN_ROOT/forecasting}")"
  ANOMALY_OUTPUT_ROOT="$(absolute_path "${ANOMALY_OUTPUT_ROOT:-$RUN_ROOT/anomaly_detection}")"
  CONCEPT_OUTPUT_ROOT="$(absolute_path "${CONCEPT_OUTPUT_ROOT:-$RUN_ROOT/concept_drift}")"
  STRUCTURED_CLASSIFICATION_OUTPUT_ROOT="$(absolute_path "${STRUCTURED_CLASSIFICATION_OUTPUT_ROOT:-$RUN_ROOT/structured_sampling/classification}")"
  STRUCTURED_FORECASTING_OUTPUT_ROOT="$(absolute_path "${STRUCTURED_FORECASTING_OUTPUT_ROOT:-$RUN_ROOT/structured_sampling/forecasting}")"
  SPAS_OUTPUT_ROOT="$(absolute_path "${SPAS_OUTPUT_ROOT:-$RUN_ROOT/SPAS}")"
  # SPAS preprocessing writes perturbed CSV files here, outside the source data.
  SPAS_DATA_ROOT="$(absolute_path "${SPAS_DATA_ROOT:-$SPAS_OUTPUT_ROOT/noised_data}")"
  # ====================== END USER PATH SETTINGS ======================
}

configure_task_paths() {
  # Task-specific path overrides belong here as well, not in the small wrappers.
  case "$1" in
    classification)
      OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-$CLASSIFICATION_OUTPUT_ROOT}")"
      RESULT_ROOT="$(absolute_path "${RESULT_ROOT:-$OUTPUT_ROOT/results}")"
      LOG_ROOT="$(absolute_path "${LOG_ROOT:-$OUTPUT_ROOT/logs}")"
      CHECKPOINT_ROOT="$(absolute_path "${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}")" ;;
    forecasting)
      OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-$FORECASTING_OUTPUT_ROOT}")"
      TRAFFIC_OUTPUT_ROOT="$(absolute_path "${TRAFFIC_OUTPUT_ROOT:-$OUTPUT_ROOT/traffic}")"
      DYNAMIC_OUTPUT_ROOT="$(absolute_path "${DYNAMIC_OUTPUT_ROOT:-$OUTPUT_ROOT/DynamicMTS}")"
      CHECKPOINT_ROOT="$(absolute_path "${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}")" ;;
    anomaly_detection)
      DATA_DIR="$(absolute_path "${DATA_DIR:-$SMD_ROOT}")"
      STUDY_ROOT="$(absolute_path "${STUDY_ROOT:-${OUTPUT_ROOT:-$ANOMALY_OUTPUT_ROOT}}")"
      local stage
      case "${SPLIT:-confirmation}" in
        confirmation) stage=confirmation ;; development) stage=screen ;;
        *) fail "SPLIT must be confirmation or development." ;;
      esac
      OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR:-$STUDY_ROOT/$stage/L250_s130}")" ;;
    concept_drift)
      OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-$CONCEPT_OUTPUT_ROOT}")"
      RESULT_ROOT="$(absolute_path "${RESULT_ROOT:-$OUTPUT_ROOT/results}")"
      CHECKPOINT_ROOT="$(absolute_path "${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}")" ;;
    structured_classification)
      OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-$STRUCTURED_CLASSIFICATION_OUTPUT_ROOT}")" ;;
    structured_forecasting)
      OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-$STRUCTURED_FORECASTING_OUTPUT_ROOT}")" ;;
    spas_classification|spas_prepare)
      OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-$SPAS_OUTPUT_ROOT}")" ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: bash scripts/launch_common.sh TASK [--dry-run] [--resume] [--show-paths]
Tasks:
  classification             HAR / USC; DLinear and PatchTST
  forecasting                LargeST / DynamicMTS; DLinear and PatchTST
  anomaly_detection          SMD privacy accounting, training, and summary
  concept_drift              LargeST / USC training and covariance analysis
  structured_classification  HAR / USC structured sampling with replacement
  structured_forecasting     LargeST / DynamicMTS structured sampling
  spas_prepare               Generate perturbed HAR / USC training CSV files
  spas_classification        Train on the perturbed HAR / USC CSV files
  all                        Run the first four tasks sequentially

Edit USER PATH SETTINGS at the top of this file, or export its variables.
DATA_ROOT holds preprocessed datasets; RUN_ROOT holds generated output.
--show-paths prints resolved settings and exits without creating directories.
--dry-run prints commands without reading data, running Python, or writing output.
--resume is supported by classification, forecasting, anomaly_detection,
concept_drift, and all. Other tasks reject existing output directories.

Common overrides: PYTHON, DEVICE=auto|cpu|cuda:N, DATASETS, MODELS,
SEEDS, SEED_START=42, NUM_RUNS=30, EPSILONS (or EPSILON_LIST).
Classification/forecasting: METHODS="baseline stratified poisson",
SAMPLING_STRIDE (100 / 50), LR, EPOCHS, BATCH_SIZE, OUTPUT_ROOT,
RESULT_ROOT, LOG_ROOT, CHECKPOINT_ROOT. TRAIN_PATH/TEST_PATH require one dataset.
Anomaly detection: DATA_DIR, SPLIT=confirmation|development, CONFIG, OUTPUT_DIR.
Concept drift: RESULT_ROOT, CHECKPOINT_ROOT, TRAFFIC_DEVICE, USC_DEVICE.
Structured sampling: MODEL_LIST, SAMPLE_STRIDE_LIST, GPU_LIST, CACHE_DIR,
RUNTIME_TMP_DIR. A single CACHE_DIR override requires one selected dataset.
SPAS: DATASETS="HAR USC", SPAS_DATA_ROOT, EPSILON=1.0, TRAIN_INDEX=0,
MODEL=DLinear, NUM_RUNS=30 (preprocessing). Paths with spaces are supported.
Jobs run sequentially in the foreground; no fixed Conda environment is required.
EOF
}

show_paths() {
  local name
  for name in PROJECT_ROOT DATA_ROOT RUN_ROOT CACHE_ROOT RUNTIME_TMP_ROOT PYTHON \
    HAR_ROOT USC_ROOT TRAFFIC_ROOT TRAFFIC_DATA_PATH DYNAMIC_ROOT DYNAMIC_DATA_PATH \
    SMD_ROOT CLASSIFICATION_OUTPUT_ROOT FORECASTING_OUTPUT_ROOT ANOMALY_OUTPUT_ROOT \
    CONCEPT_OUTPUT_ROOT STRUCTURED_CLASSIFICATION_OUTPUT_ROOT \
    STRUCTURED_FORECASTING_OUTPUT_ROOT SPAS_OUTPUT_ROOT SPAS_DATA_ROOT \
    OUTPUT_ROOT RESULT_ROOT LOG_ROOT CHECKPOINT_ROOT TRAFFIC_OUTPUT_ROOT \
    DYNAMIC_OUTPUT_ROOT DATA_DIR STUDY_ROOT OUTPUT_DIR; do
    [[ ! -v "$name" ]] || printf '%s=%s\n' "$name" "${!name}"
  done
}


DRY_RUN=0
RESUME=0
SHOW_PATHS=0

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

parse_launch_args() {
  while (( $# )); do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --resume) RESUME=1 ;;
      --show-paths) SHOW_PATHS=1 ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; fail "Unknown argument: $1" ;;
    esac
    shift
  done
}

# Resolve paths before any launcher changes directory, including missing paths
# in dry runs. PYTHON is one executable, not a string of shell commands.
# This normalizes a path; it does not create directories or check dataset files.
absolute_path() { realpath -m -- "$1"; }

init_python() {
  PYTHON="${PYTHON:-python}"
  if [[ "$PYTHON" == */* ]]; then
    PYTHON="$(absolute_path "$PYTHON")"
  elif (( ! DRY_RUN )); then
    PYTHON="$(command -v -- "$PYTHON")" || fail "Python not found; activate your environment or set PYTHON."
  fi
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
}

require_file() {
  (( DRY_RUN )) || [[ -f "$1" && -r "$1" ]] || fail "File not readable: $1"
}

positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer; got: $2"
}

positive_number() {
  [[ "$2" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] &&
    awk -v value="$2" 'BEGIN { exit !(value > 0) }' || fail "$1 must be positive; got: $2"
}

init_sweep_seeds() {
  local count="${NUM_RUNS:-${ITR:-30}}" start="${SEED_START:-42}" offset seed
  positive_integer NUM_RUNS "$count"
  [[ "$start" =~ ^[0-9]+$ ]] || fail "SEED_START must be nonnegative."
  seeds=()
  if [[ -n "${SEEDS:-}" ]]; then
    read -r -a seeds <<<"$SEEDS"
  else
    for ((offset = 0; offset < count; offset++)); do seeds+=("$((start + offset))"); done
  fi
  (( ${#seeds[@]} )) || fail "SEEDS must not be empty."
  for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || fail "Invalid seed: $seed"
  done
}

validate_sweep() {
  (( ${#datasets[@]} && ${#models[@]} && ${#methods[@]} && ${#epsilons[@]} && ${#strides[@]} )) ||
    fail "Dataset, model, method, epsilon and stride lists must not be empty."
  local epsilon
  for epsilon in "${epsilons[@]}"; do
    positive_number EPSILON "$epsilon"
  done
  positive_integer BATCH_SIZE "${BATCH_SIZE:-50}"
  [[ "${NUM_WORKERS:-10}" =~ ^[0-9]+$ ]] || fail "NUM_WORKERS must be nonnegative."
  [[ -z "${EPOCHS:-}" ]] || positive_integer EPOCHS "$EPOCHS"
}

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

run_logged() {
  local log="$1"
  shift
  print_command "$@"
  if (( ! DRY_RUN )); then
    "$@" 2>&1 | tee -a "$log"
  fi
}

# Classification/forecasting write one numeric summary per seed.
valid_result() {
  [[ -s "$1" ]] && awk '
    NF < 2 { exit 1 }
    { for (i = 1; i <= NF; i++)
        if ($i !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) exit 1
      seen = 1 }
    END { if (!seen) exit 1 }
  ' "$1"
}

run_training_job() {
  local job_dir="$1" result_file="$2" log_file="$3" checkpoint_dir="$4"
  shift 4
  printf '[JOB] %s\n' "$result_file"
  printf '[LOG] %s\n[CHECKPOINTS] %s\n' "$log_file" "$checkpoint_dir"
  if (( DRY_RUN )); then
    print_command "$@"
    return
  fi
  if [[ -e "$job_dir" || -e "$result_file" || -e "$log_file" ]]; then
    (( RESUME )) || fail "Output exists: $result_file (use --resume or new output paths)."
    if [[ -f "$job_dir/command.args" ]]; then
      cmp -s "$job_dir/command.args" <(printf '%s\0' "$@") ||
        fail "Configuration changed in $job_dir; choose new output paths."
    elif valid_result "$result_file"; then
      printf '[SKIP] Existing VLDB result (no launcher signature): %s\n' "$result_file"
      return
    else
      fail "Existing incomplete output has no launcher signature: $result_file; choose new output paths."
    fi
    if [[ -f "$job_dir/completed" ]] && valid_result "$result_file"; then
      printf '[SKIP] Completed: %s\n' "$result_file"
      return
    fi
    printf '[RESTART] Incomplete seed; training will start again.\n'
  fi
  mkdir -p "$job_dir" "$(dirname -- "$result_file")" "$(dirname -- "$log_file")" "$checkpoint_dir"
  rm -f -- "$job_dir/completed"
  if [[ -e "$result_file" ]]; then
    mv -- "$result_file" "$job_dir/results.interrupted.$(date -u +%Y%m%dT%H%M%S).$$.txt"
  fi
  printf '%s\0' "$@" >"$job_dir/command.args"
  print_command "$@" >"$job_dir/command.sh"
  # Keep legacy runners' relative auxiliary output inside this job directory.
  (cd "$job_dir"; "$@") 2>&1 | tee -a "$log_file"
  valid_result "$result_file" || fail "Missing or non-finite results: $result_file"
  touch "$job_dir/completed"
}


launch_classification() (
SCRIPT_DIR="$PROJECT_ROOT/classification"
read -r -a datasets <<<"${DATASETS:-HAR USC}"
read -r -a models <<<"${MODELS:-DLinear PatchTST}"
read -r -a methods <<<"${METHODS:-baseline stratified poisson}"
read -r -a epsilons <<<"${EPSILON_LIST:-${EPSILONS:-0.1 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6}}"
SEQ_LEN="${SEQ_LEN:-200}"
init_sweep_seeds
read -r -a strides <<<"${SAMPLING_STRIDES:-${SAMPLING_STRIDE:-100}}"
positive_integer SEQ_LEN "$SEQ_LEN"
for stride in "${strides[@]}"; do positive_integer SAMPLING_STRIDE "$stride"; done
validate_sweep
if [[ -n "${TRAIN_PATH:-}${TEST_PATH:-}" && ${#datasets[@]} != 1 ]]; then
  fail "TRAIN_PATH/TEST_PATH require exactly one DATASETS entry."
fi
for dataset in "${datasets[@]}"; do
  case "$dataset" in HAR) root="$HAR_ROOT" ;; USC) root="$USC_ROOT" ;; *) fail "Unknown dataset: $dataset" ;; esac
  train="$(absolute_path "${TRAIN_PATH:-$root/training_data.csv}")"
  require_file "$train"
  require_file "$(dirname -- "$train")/validation_data.csv"
  require_file "$(absolute_path "${TEST_PATH:-$root/testing_data.csv}")"
done
for model in "${models[@]}"; do
  case "$model" in DLinear|DLinear_Stable|PatchTST) ;; *) fail "Unknown model: $model" ;; esac
done
for method in "${methods[@]}"; do
  case "$method" in baseline|poisson|stratified) ;; *) fail "Unknown method: $method" ;; esac
done
gpu="${GPU:-0}"
case "${DEVICE:-auto}" in
  auto) ;;
  cpu) export CUDA_VISIBLE_DEVICES=-1; gpu=0 ;;
  cuda:*) gpu="${DEVICE#cuda:}" ;;
  *) fail "DEVICE must be auto, cpu or cuda:<index>." ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU must be a nonnegative CUDA index."
device_env=(env)
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
  device_env+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
else
  device_env+=(-u CUDA_VISIBLE_DEVICES)
fi

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == HAR ]]; then root="$HAR_ROOT"; channels=42; classes=8
  else root="$USC_ROOT"; channels=6; classes=12; fi
  train="$(absolute_path "${TRAIN_PATH:-$root/training_data.csv}")"
  test="$(absolute_path "${TEST_PATH:-$root/testing_data.csv}")"
  for model in "${models[@]}"; do
    case "$dataset/$model" in
      HAR/DLinear|HAR/DLinear_Stable) lr=0.02; epochs="${DLINEAR_EPOCHS:-100}"; step=30 ;;
      USC/DLinear|USC/DLinear_Stable) lr=0.1; epochs="${DLINEAR_EPOCHS:-100}"; step=30 ;;
      */PatchTST) lr=0.01; epochs="${PATCHTST_EPOCHS:-8}"; step=20 ;;
    esac
    lr="${LR:-$lr}"
    for stride in "${strides[@]}"; do
      for epsilon in "${epsilons[@]}"; do
        for seed in "${seeds[@]}"; do
          for method in "${methods[@]}"; do
            case "$method" in
              baseline) runner=run_DPSGD_TSC.py ;;
              poisson) runner=run_poisson_TSC.py ;;
              stratified) runner=run_spaced_TSC.py ;;
            esac
            stem="${dataset}_${model}_${method}_ss${stride}_len${SEQ_LEN}_eps${epsilon}"
            result_file="$RESULT_ROOT/$model/$method/${stem}_lr${lr}_clip${CLIPPING_NORM:-1}_seed${seed}.txt"
            log_file="$LOG_ROOT/$model/$method/${stem}_seed${seed}.log"
            checkpoint_dir="$CHECKPOINT_ROOT/$model/$method"
            job="$LOG_ROOT/$model/$method/.launch_state/${stem}_lr${lr}_clip${CLIPPING_NORM:-1}_seed${seed}"
            run_training_job "$job" "$result_file" "$log_file" "$checkpoint_dir" "${device_env[@]}" \
              "$PYTHON" -u "$SCRIPT_DIR/$runner" \
              --is_training 1 --model_id "${dataset}_${model}_${method}_sl${SEQ_LEN}_ss${stride}_seed${seed}" --model "$model" \
              --seq_len "$SEQ_LEN" --sampling_stride "$stride" --num_kernels "${NUM_KERNELS:-1000}" \
              --enc_in "$channels" --num_classes "$classes" --batch_size "${BATCH_SIZE:-50}" \
              --num_workers "${NUM_WORKERS:-10}" --train_epochs "${EPOCHS:-$epochs}" --itr 1 --seed "$seed" \
              --learning_rate "$lr" --lr_step_size "${LR_STEP_SIZE:-$step}" --lr_gamma "${LR_GAMMA:-0.9}" \
              --stride "${PATCH_STRIDE:-8}" --revin "${REVIN:-0}" --affine 0 \
              --gpu "$gpu" --dp_sigma "${DP_SIGMA:-5}" --dp_delta "${DP_DELTA:-1e-5}" \
              --clipping_norm "${CLIPPING_NORM:-1}" --privacy_budget_limit "$epsilon" --w "${W:-0.005}" \
              --label_mode sequence --train_path "$train" --test_path "$test" \
              --result_file "$result_file" --checkpoints "$checkpoint_dir"
          done
        done
      done
    done
  done
done
printf 'Output: %s\n' "$OUTPUT_ROOT"

)


launch_forecasting() (
SCRIPT_DIR="$PROJECT_ROOT/forecasting"
read -r -a datasets <<<"${DATASETS:-LargestTS DynamicMTS}"
read -r -a models <<<"${MODELS:-DLinear PatchTST}"
read -r -a methods <<<"${METHODS:-baseline stratified poisson}"
full_epsilons="0.1 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6"
read -r -a epsilons <<<"${EPSILON_LIST:-${EPSILONS:-$full_epsilons}}"
SEQ_LEN="${SEQ_LEN:-80}"
PRED_LEN="${PRED_LEN:-20}"
init_sweep_seeds
positive_integer SEQ_LEN "$SEQ_LEN"
positive_integer PRED_LEN "$PRED_LEN"
read -r -a strides <<<"${SAMPLING_STRIDES:-${SAMPLING_STRIDE:-50}}"
for stride in "${strides[@]}"; do positive_integer SAMPLING_STRIDE "$stride"; done
validate_sweep
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    LargestTS|LargeST|traffic) root="$TRAFFIC_ROOT"; file="${TRAFFIC_DATA_PATH:-traffic.csv}" ;;
    DynamicMTS) root="$DYNAMIC_ROOT"; file="${DYNAMIC_DATA_PATH:-dynamic_data.csv}" ;;
    *) fail "Unknown dataset: $dataset" ;;
  esac
  [[ "$file" == /* ]] || file="$root/$file"
  require_file "$file"
done
for model in "${models[@]}"; do
  case "$model" in DLinear|PatchTST) ;; *) fail "Unknown model: $model" ;; esac
done
for method in "${methods[@]}"; do
  case "$method" in baseline|stratified|poisson) ;; *) fail "Unknown method: $method" ;; esac
done
gpu="${GPU:-0}"
case "${DEVICE:-auto}" in
  auto) ;;
  cpu) gpu=-1 ;; # The Python entry point sets CUDA_VISIBLE_DEVICES from --devices.
  cuda:*) gpu="${DEVICE#cuda:}" ;;
  *) fail "DEVICE must be auto, cpu or cuda:<index>." ;;
esac
[[ "$gpu" =~ ^[0-9]+$ || ( "$gpu" == -1 && "${DEVICE:-auto}" == cpu ) ]] || fail "Invalid GPU index."

for selected_dataset in "${datasets[@]}"; do
  case "$selected_dataset" in
    LargestTS|LargeST|traffic)
      dataset=LargestTS; mid=traffic; output_root="$TRAFFIC_OUTPUT_ROOT"
      root="$TRAFFIC_ROOT"; file="${TRAFFIC_DATA_PATH:-traffic.csv}"; channels=371 ;;
    DynamicMTS)
      dataset=DynamicMTS; mid=dynamic; output_root="$DYNAMIC_OUTPUT_ROOT"
      root="$DYNAMIC_ROOT"; file="${DYNAMIC_DATA_PATH:-dynamic_data.csv}"; channels=17 ;;
  esac
  for model in "${models[@]}"; do
    case "$dataset/$model" in
      LargestTS/DLinear) lr=0.02; default_epsilons="$full_epsilons" ;;
      LargestTS/PatchTST) lr=0.005; default_epsilons=0.1 ;;
      DynamicMTS/DLinear) lr=0.005; default_epsilons="$full_epsilons" ;;
      DynamicMTS/PatchTST) lr=0.0015; default_epsilons="$full_epsilons" ;;
    esac
    lr="${LR:-$lr}"
    read -r -a epsilons <<<"${EPSILON_LIST:-${EPSILONS:-$default_epsilons}}"
    for stride in "${strides[@]}"; do
      for epsilon in "${epsilons[@]}"; do
        for seed in "${seeds[@]}"; do
          for method in "${methods[@]}"; do
            case "$method" in
              baseline) runner=run_Exp.py; suffix="" ;;
              stratified) runner=run_Exp_stratification.py; suffix=_strat ;;
              poisson) runner=run_Exp_spaced_sampling_poisson.py; suffix=_poisson ;;
            esac
            stem="${mid}${suffix}_ss${stride}_len${SEQ_LEN}_pred${PRED_LEN}_eps${epsilon}"
            result_file="$output_root/results/${model}${suffix}/${stem}_lr${lr}_clip${CLIPPING_NORM:-1}_seed${seed}.txt"
            log_file="$output_root/logs/${model}${suffix}/${stem}_seed${seed}.log"
            job="$output_root/logs/${model}${suffix}/.launch_state/${stem}_lr${lr}_clip${CLIPPING_NORM:-1}_seed${seed}"
            run_training_job "$job" "$result_file" "$log_file" "$CHECKPOINT_ROOT" "$PYTHON" -u "$SCRIPT_DIR/$runner" \
              --is_training 1 --root_path "$root" --data_path "$file" \
              --model_id "${mid}_${SEQ_LEN}_${PRED_LEN}_ss${stride}" --model "$model" --data custom \
              --des "$method" --features M --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
              --sampling_stride "$stride" --stride "${PATCH_STRIDE:-8}" --revin "${REVIN:-0}" \
              --enc_in "$channels" --c_out "$channels" --train_epochs "${EPOCHS:-100}" \
              --patience "${PATIENCE:-5}" --micro_batch_size "${MICRO_BATCH_SIZE:-50}" \
              --num_workers "${NUM_WORKERS:-10}" --itr 1 --seed "$seed" --batch_size "${BATCH_SIZE:-50}" \
              --learning_rate "$lr" --dp_sigma "${DP_SIGMA:-5}" --dp_delta "${DP_DELTA:-1e-5}" \
              --privacy_budget_limit "$epsilon" --w "${W:-0.005}" --devices "$gpu" \
              --gpu_reserve_mb "${GPU_RESERVE_MB:-0}" \
              --clipping_norm "${CLIPPING_NORM:-1}" --result_file "$result_file" --checkpoints "$CHECKPOINT_ROOT"
          done
        done
      done
    done
  done
done
printf 'Output: %s and %s\n' "$TRAFFIC_OUTPUT_ROOT" "$DYNAMIC_OUTPUT_ROOT"

)


launch_anomaly_detection() (
SCRIPT_DIR="$PROJECT_ROOT/anomaly_detection"
SPLIT="${SPLIT:-confirmation}"
case "$SPLIT" in
  confirmation) stage=confirmation ;;
  development) stage=screen ;;
  *) fail "SPLIT must be confirmation or development." ;;
esac
positive_integer NUM_THREADS "${NUM_THREADS:-1}"
config_args=(--split "$SPLIT")
if [[ -n "${CONFIG:-}" ]]; then
  CONFIG="$(absolute_path "$CONFIG")"
  require_file "$CONFIG"
  config_args=(--config "$CONFIG")
fi
for file in train.csv test.csv test_label.csv; do require_file "$DATA_DIR/$file"; done
if (( ! DRY_RUN )); then
  [[ ! -e "$OUTPUT_DIR" ]] || (( RESUME )) || fail "OUTPUT_DIR exists; use --resume or choose a new directory."
  mkdir -p "$OUTPUT_DIR"
fi
cd "$PROJECT_ROOT"
common=(--data-dir "$DATA_DIR" "${config_args[@]}" --output-dir "$OUTPUT_DIR")
resume_args=()
(( ! RESUME )) || resume_args=(--resume)
run_logged "$OUTPUT_DIR/launch.log" "$PYTHON" -u -m anomaly_detection account "${common[@]}"
run_logged "$OUTPUT_DIR/launch.log" "$PYTHON" -u -m anomaly_detection sweep "${common[@]}" \
  --device "${DEVICE:-auto}" --num-threads "${NUM_THREADS:-1}" --save-scores "${resume_args[@]}"
run_logged "$OUTPUT_DIR/launch.log" "$PYTHON" -u -m anomaly_detection summarize --output-dir "$OUTPUT_DIR"
printf 'Output: %s\n' "$OUTPUT_DIR"

)

run_covariance_dataset() (
  set -euo pipefail
  local dataset="$1"
  local device="$2"
  local output="${RESULT_ROOT}/${dataset}"
  local training_output="${CHECKPOINT_ROOT}/${dataset}"
  local checkpoint="${training_output}/checkpoint"
  local training_complete=0
  if (( ! DRY_RUN )); then
  mkdir -p "${output}" "${training_output}"
  if [[ "$MODE" == --resume ]]; then
    if [[ -f "${checkpoint}/checkpoint.pth" &&
          -f "${checkpoint}/config.json" &&
          -f "${checkpoint}/metrics.json" ]]; then
      training_complete=1
    fi
    local archive="${output}/interrupted_$(date -u +%Y%m%dT%H%M%S)"
    mkdir -p "$archive"
    local filename
    for filename in run.log covariance.log status.txt; do
      if [[ -f "${output}/${filename}" ]]; then
        mv "${output}/${filename}" "$archive/"
      fi
    done
    if (( training_complete == 0 )); then
      if [[ -f "${training_output}/train.log" ]]; then
        mv "${training_output}/train.log" "$archive/"
      fi
      if [[ -d "${checkpoint}" ]]; then
        mv "${checkpoint}" "$archive/"
      fi
    fi
  fi
  exec >"${output}/run.log" 2>&1
  trap 'code=$?; if (( code != 0 )); then printf "failed exit=%s time=%s\n" "$code" "$(date --iso-8601=seconds)" >"${output}/status.txt"; fi' EXIT
  printf 'training %s\n' "$(date --iso-8601=seconds)" >"${output}/status.txt"
  printf 'dataset=%s device=%s started=%s\n' "$dataset" "$device" "$(date --iso-8601=seconds)"
  fi
  local architecture=(--model PatchTST --seq-len 160 --batch-size 50
    --patience 5 --learning-rate 1e-4 --seed 42 --num-workers 0
    --device "${device}" --e-layers 2 --n-heads 8 --d-model 128
    --d-ff 512 --patch-len 16 --patch-stride 8 --padding-patch end
    --checkpoint-dir "${checkpoint}")
  local task=classification
  local data_args=()
  if [[ "$dataset" == traffic ]]; then
    task=forecasting
    data_args=(--root-path "${TRAFFIC_ROOT}" --data-path "${TRAFFIC_DATA_PATH:-traffic.csv}"
      --target "${TRAFFIC_TARGET:-1211352}" --features M)
  fi
  if (( training_complete == 1 )); then
    echo "Reusing completed training checkpoint: ${checkpoint}"
  elif [[ "$dataset" == traffic ]]; then
    run_logged "${training_output}/train.log" "$PYTHON" -u "${PACKAGE_ROOT}/model_code/train_forecasting.py" \
      "${architecture[@]}" "${data_args[@]}" --pred-len 20 --label-len 0 \
      --enc-in 371 --sample-stride 180 --epochs 50 \
      --dropout 0 --fc-dropout 0 --head-dropout 0 \
      --revin 1 --affine 0 --subtract-last 0 --decomposition 0
  else
    task=classification
    run_logged "${training_output}/train.log" "$PYTHON" -u "${PACKAGE_ROOT}/classification_model_code/train_classification.py" \
      "${architecture[@]}" --enc-in 6 --num-classes 12 --epochs 30 \
      --train-path "${USC_ROOT}/training_data.csv" \
      --val-path "${USC_ROOT}/validation_data.csv" \
      --test-path "${USC_ROOT}/testing_data.csv" \
      --label-column activity
  fi
  if (( ! DRY_RUN )); then printf 'covariance %s\n' "$(date --iso-8601=seconds)" >"${output}/status.txt"; fi
  run_logged "${output}/covariance.log" "$PYTHON" -u -m concept_drift.analyze --task "$task" --model PatchTST \
    --checkpoint "${checkpoint}/checkpoint.pth" "${data_args[@]}" \
    --split train --device "${device}" --methods baseline poisson \
    --poisson-lambda auto --dp-sigma 5 --dp-delta 1e-5 \
    --privacy-budget-limit 6 --w 0.005 \
    --batch-size 60 --candidate-stride 1 --seed 42 --trials 10000 \
    --bootstrap 0 --projection countsketch --projection-dim 128 \
    --projection-seed 2026 --projection-seed-count 30 --progress-every 100 \
    --output-dir "${output}/covariance"
  if (( ! DRY_RUN )); then
  "$PYTHON" - "$output" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
report = json.loads((output / "covariance/report.json").read_text())
assert report["experiment"]["batch_size_K"] == 60
assert len(report["projection_seed_runs"]) == 30
assert set(report["sampling_comparison"]) == {"baseline", "poisson"}
for method in ("baseline", "poisson"):
    assert math.isfinite(report["unprojected_trace"][method]["trace"])
    for run in report["projection_seed_runs"]:
        for metric in ("trace", "frobenius", "spectral"):
            assert math.isfinite(run["sampling_comparison"][method]["gradient_covariance"][metric])
print("Verified raw trace and all projected trace/Frobenius/spectral results")
PY
  printf 'completed %s\n' "$(date --iso-8601=seconds)" >"${output}/status.txt"
  fi
)

launch_concept_drift() (
  PACKAGE_ROOT="$PROJECT_ROOT/concept_drift"
  MODE=new
  (( ! RESUME )) || MODE=--resume
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  require_file "$(dataset_file "$TRAFFIC_ROOT" "$TRAFFIC_DATA_PATH")"
  for file in training_data.csv validation_data.csv testing_data.csv; do require_file "$USC_ROOT/$file"; done
  if (( ! DRY_RUN )); then
    [[ ( ! -e "$RESULT_ROOT" && ! -e "$CHECKPOINT_ROOT" ) || "$MODE" == --resume ]] ||
      fail "Result/checkpoint root exists; use --resume or new output paths."
    mkdir -p "$RESULT_ROOT"
  fi
  cd "$PROJECT_ROOT"
  run_covariance_dataset traffic "${TRAFFIC_DEVICE:-${DEVICE:-auto}}"
  run_covariance_dataset usc "${USC_DEVICE:-${DEVICE:-auto}}"
  run_logged "$RESULT_ROOT/summary.log" "$PYTHON" -m concept_drift.summarize_results \
    --result-root "$RESULT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"
)

# Resolve a dataset filename while allowing an explicit absolute CSV path.
dataset_file() {
  if [[ "$2" == /* ]]; then absolute_path "$2"; else absolute_path "$1/$2"; fi
}

new_output_directory() {
  if (( ! DRY_RUN )); then
    [[ ! -e "$1" ]] || fail "Output already exists: $1. Choose a new OUTPUT_ROOT."
    mkdir -p "$1/logs"
  fi
}

classification_dataset() {
  case "$1" in
    HAR) dataset_root="$HAR_ROOT"; channels=42; classes=8 ;;
    USC) dataset_root="$USC_ROOT"; channels=6; classes=12 ;;
    *) fail "Unknown classification dataset: $1 (choose HAR or USC)." ;;
  esac
  train="$(absolute_path "${TRAIN_PATH:-$dataset_root/training_data.csv}")"
  validation="$(absolute_path "${VALIDATION_PATH:-$dataset_root/validation_data.csv}")"
  test="$(absolute_path "${TEST_PATH:-$dataset_root/testing_data.csv}")"
}

init_structured_sweep() {
  init_sweep_seeds
  read -r -a epsilons <<<"${EPSILON_LIST:-${EPSILONS:-0.1 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6}}"
  methods=(structured)
  validate_sweep
  for stride in "${strides[@]}"; do positive_integer SAMPLE_STRIDE "$stride"; done
  for model in "${models[@]}"; do
    case "$model" in DLinear|PatchTST) ;; *) fail "Unknown model: $model" ;; esac
  done
  read -r -a physical_gpus <<<"${GPU_LIST:-}"
  for gpu in "${physical_gpus[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || fail "Invalid GPU_LIST entry: $gpu"; done
  if [[ -n "${CACHE_DIR:-}${TRAIN_PATH:-}${VALIDATION_PATH:-}${TEST_PATH:-}" && ${#datasets[@]} != 1 ]]; then
    fail "Split path and CACHE_DIR overrides require exactly one dataset."
  fi
}

structured_device() {
  device_args=(env)
  device="${DEVICE:-auto}"
  if [[ "$device" != cpu && ${#physical_gpus[@]} -gt 0 ]]; then
    device_args+=("CUDA_VISIBLE_DEVICES=${physical_gpus[$(($1 % ${#physical_gpus[@]}))]}")
    device=cuda:0
  fi
}

launch_structured_classification() (
  local script_dir="$PROJECT_ROOT/structured_sampling/classification"
  read -r -a datasets <<<"${DATASETS:-${DATASET:-HAR USC}}"
  read -r -a models <<<"${MODEL_LIST:-${MODELS:-DLinear PatchTST}}"
  read -r -a strides <<<"${SAMPLE_STRIDE_LIST:-${SAMPLING_STRIDE:-100}}"
  init_structured_sweep
  # Validate every selected split before creating any output.
  for dataset in "${datasets[@]}"; do
    classification_dataset "$dataset"
    for file in "$train" "$validation" "$test"; do require_file "$file"; done
  done
  local job_index=0
  for dataset in "${datasets[@]}"; do
    classification_dataset "$dataset"
    local output="$OUTPUT_ROOT/$dataset"
    local cache="$(absolute_path "${CACHE_DIR:-$CACHE_ROOT/structured_classification/$dataset}")"
    local runtime_tmp="$(absolute_path "${RUNTIME_TMP_DIR:-$RUNTIME_TMP_ROOT/structured_classification/$dataset}")"
    new_output_directory "$output"
    if (( ! DRY_RUN )); then mkdir -p "$runtime_tmp"; fi
    export TMPDIR="$runtime_tmp"
    run_logged "$output/logs/prepare.log" "$PYTHON" "$script_dir/prepare_n1_data.py" \
      --dataset "$dataset" --train-path "$train" --validation-path "$validation" \
      --test-path "$test" --cache-dir "$cache"
    for model in "${models[@]}"; do
      for stride in "${strides[@]}"; do
        for epsilon in "${epsilons[@]}"; do
          structured_device "$job_index"
          local stride_root="$output/sample_stride_$stride"
          run_logged "$output/logs/${model}_ss${stride}_eps${epsilon}.log" \
            "${device_args[@]}" "$PYTHON" "$script_dir/run_n1_epsilon_sweep.py" \
            --dataset "$dataset" --train-path "$train" --validation-path "$validation" --test-path "$test" \
            --data-cache "$cache" --output-root "$stride_root" --models "$model" \
            --epsilons "$epsilon" --seeds "${seeds[@]}" --device "$device" --sample-stride "$stride" \
            --seq-len "${SEQ_LEN:-200}" --enc-in "${ENC_IN:-$channels}" --num-classes "${NUM_CLASSES:-$classes}" \
            --batch-size "${BATCH_SIZE:-50}" --dlinear-learning-rate "${DLINEAR_LEARNING_RATE:-0.02}" \
            --patchtst-learning-rate "${PATCHTST_LEARNING_RATE:-0.01}" \
            --summary-file "$stride_root/$model/epsilon_$epsilon/summary.json"
          job_index=$((job_index + 1))
        done
      done
    done
    for stride in "${strides[@]}"; do
      run_logged "$output/logs/summary_ss${stride}.log" "$PYTHON" "$script_dir/summarize_n1_results.py" \
        --output-root "$output/sample_stride_$stride"
    done
  done
)

forecasting_dataset() {
  case "$1" in
    LargestTS|LargeST|traffic)
      label=traffic; dataset_root="$TRAFFIC_ROOT"; data_path="$TRAFFIC_DATA_PATH"
      channels="${TRAFFIC_EXPECTED_NUM_VARIABLES:-371}"
      cache="${CACHE_DIR:-${TRAFFIC_CACHE_DIR:-$CACHE_ROOT/structured_forecasting/traffic}}"
      runtime_tmp="${RUNTIME_TMP_DIR:-${TRAFFIC_RUNTIME_TMP_DIR:-$RUNTIME_TMP_ROOT/structured_forecasting/traffic}}"
      output="${TRAFFIC_OUTPUT_ROOT:-$OUTPUT_ROOT/traffic}"
      context_len="${TRAFFIC_CONTEXT_LEN:-80}"; forecast_len="${TRAFFIC_FORECAST_LEN:-20}"
      dlinear_lr="${TRAFFIC_DLINEAR_LEARNING_RATE:-0.02}"; patchtst_lr="${TRAFFIC_PATCHTST_LEARNING_RATE:-0.005}" ;;
    DynamicMTS|dynamic)
      label=DynamicMTS; dataset_root="$DYNAMIC_ROOT"; data_path="$DYNAMIC_DATA_PATH"
      channels="${DYNAMIC_EXPECTED_NUM_VARIABLES:-17}"
      cache="${CACHE_DIR:-${DYNAMIC_CACHE_DIR:-$CACHE_ROOT/structured_forecasting/DynamicMTS}}"
      runtime_tmp="${RUNTIME_TMP_DIR:-${DYNAMIC_RUNTIME_TMP_DIR:-$RUNTIME_TMP_ROOT/structured_forecasting/DynamicMTS}}"
      output="${DYNAMIC_OUTPUT_ROOT:-$OUTPUT_ROOT/DynamicMTS}"
      context_len="${DYNAMIC_CONTEXT_LEN:-80}"; forecast_len="${DYNAMIC_FORECAST_LEN:-20}"
      dlinear_lr="${DYNAMIC_DLINEAR_LEARNING_RATE:-0.005}"; patchtst_lr="${DYNAMIC_PATCHTST_LEARNING_RATE:-0.0015}" ;;
    *) fail "Unknown forecasting dataset: $1" ;;
  esac
  data_path="$(dataset_file "$dataset_root" "$data_path")"
  output="$(absolute_path "$output")"; cache="$(absolute_path "$cache")"; runtime_tmp="$(absolute_path "$runtime_tmp")"
}

launch_structured_forecasting() (
  local script_dir="$PROJECT_ROOT/structured_sampling/forecasting"
  read -r -a datasets <<<"${DATASETS:-LargestTS DynamicMTS}"
  read -r -a models <<<"${MODEL_LIST:-${MODELS:-PatchTST}}"
  read -r -a strides <<<"${SAMPLE_STRIDE_LIST:-${SAMPLING_STRIDE:-50}}"
  init_structured_sweep
  for dataset in "${datasets[@]}"; do
    forecasting_dataset "$dataset"
    require_file "$data_path"
    if (( ! DRY_RUN )); then
      observed="$(awk -F, 'NR == 1 {print NF - 1; exit}' "$data_path")"
      [[ "$observed" == "$channels" ]] || fail "$label requires $channels feature columns; found $observed."
    fi
  done
  local job_index=0
  for dataset in "${datasets[@]}"; do
    forecasting_dataset "$dataset"
    new_output_directory "$output"
    if (( ! DRY_RUN )); then mkdir -p "$runtime_tmp"; fi
    export TMPDIR="$runtime_tmp"
    run_logged "$output/logs/prepare.log" "$PYTHON" "$script_dir/prepare_n1_data.py" \
      --root-path "$dataset_root" --data-path "$data_path" --cache-dir "$cache"
    for model in "${models[@]}"; do
      for stride in "${strides[@]}"; do
        for epsilon in "${epsilons[@]}"; do
          structured_device "$job_index"
          local stride_root="$output/sample_stride_$stride"
          run_logged "$output/logs/${model}_ss${stride}_eps${epsilon}.log" \
            "${device_args[@]}" "$PYTHON" "$script_dir/run_n1_epsilon_sweep.py" \
            --root-path "$dataset_root" --data-path "$data_path" --data-cache "$cache" \
            --output-root "$stride_root" --models "$model" --epsilons "$epsilon" --seeds "${seeds[@]}" \
            --device "$device" --num-workers "${NUM_WORKERS:-0}" --sample-stride "$stride" \
            --context-len "$context_len" --forecast-len "$forecast_len" \
            --dlinear-learning-rate "$dlinear_lr" --patchtst-learning-rate "$patchtst_lr" \
            --summary-file "$stride_root/$model/epsilon_$epsilon/summary.json"
          job_index=$((job_index + 1))
        done
      done
    done
    for stride in "${strides[@]}"; do
      run_logged "$output/logs/summary_ss${stride}.log" "$PYTHON" "$script_dir/summarize_n1_results.py" \
        --output-root "$output/sample_stride_$stride"
    done
  done
)

launch_spas_prepare() (
  read -r -a datasets <<<"${DATASETS:-HAR USC}"
  read -r -a epsilons <<<"${EPSILON_LIST:-${EPSILONS:-3 2.75 2.5 2.25 2 1.75 1.5 1.25 1 0.75 0.5}}"
  local count="${NUM_RUNS:-30}"
  positive_integer NUM_RUNS "$count"
  [[ ${#datasets[@]} -gt 0 && ${#epsilons[@]} -gt 0 ]] || fail "DATASETS and EPSILONS must not be empty."
  for epsilon in "${epsilons[@]}"; do positive_number EPSILON "$epsilon"; done
  [[ -z "${TRAIN_PATH:-}" || ${#datasets[@]} == 1 ]] || fail "TRAIN_PATH requires one dataset."
  for dataset in "${datasets[@]}"; do classification_dataset "$dataset"; require_file "$train"; done
  for dataset in "${datasets[@]}"; do
    classification_dataset "$dataset"
    local output="$SPAS_DATA_ROOT/$dataset"
    new_output_directory "$output"
    for epsilon in "${epsilons[@]}"; do
      run_logged "$output/logs/epsilon_${epsilon}.log" "$PYTHON" "$PROJECT_ROOT/SPAS/SPAS_data/process_data.py" \
        --file_path "$train" --epsilon "$epsilon" --start_i 0 --end_i "$count" --output-dir "$output"
    done
  done
)

launch_spas_classification() (
  read -r -a datasets <<<"${DATASETS:-HAR USC}"
  local epsilon="${EPSILON:-1.0}" index="${TRAIN_INDEX:-0}" model="${MODEL:-DLinear}"
  [[ ${#datasets[@]} -gt 0 ]] || fail "DATASETS must not be empty."
  positive_number EPSILON "$epsilon"
  local epsilon_tag
  epsilon_tag="$(awk -v value="$epsilon" 'BEGIN {printf "%.12g", value}')"
  [[ "$epsilon_tag" == *.* || "$epsilon_tag" == *e* ]] || epsilon_tag+=.0
  [[ "$index" =~ ^[0-9]+$ ]] || fail "TRAIN_INDEX must be nonnegative."
  [[ "$model" == DLinear || "$model" == PatchTST ]] || fail "MODEL must be DLinear or PatchTST."
  [[ -z "${TRAIN_PATH:-}${VALIDATION_PATH:-}${TEST_PATH:-}" || ${#datasets[@]} == 1 ]] || fail "Split overrides require one dataset."
  for dataset in "${datasets[@]}"; do
    classification_dataset "$dataset"
    train="$(absolute_path "${TRAIN_PATH:-$SPAS_DATA_ROOT/$dataset/epsilon_$epsilon_tag/train_noised_$index.csv}")"
    for file in "$train" "$validation" "$test"; do require_file "$file"; done
  done
  for dataset in "${datasets[@]}"; do
    classification_dataset "$dataset"
    train="$(absolute_path "${TRAIN_PATH:-$SPAS_DATA_ROOT/$dataset/epsilon_$epsilon_tag/train_noised_$index.csv}")"
    local output="$OUTPUT_ROOT/classification/$dataset/$model/epsilon_$epsilon_tag/run_$index"
    local checkpoint="$(absolute_path "${CHECKPOINT_ROOT:-$output/checkpoints}")"
    new_output_directory "$output"
    local gpu="${GPU:-0}" use_gpu=1
    case "${DEVICE:-auto}" in cpu) use_gpu=0 ;; cuda:*) gpu="${DEVICE#cuda:}" ;; esac
    # Legacy SPAS saves auxiliary files relative to its working directory.
    if (( ! DRY_RUN )); then cd "$output"; fi
    run_logged "$output/logs/train.log" "$PYTHON" "$PROJECT_ROOT/SPAS/run_longExp.py" \
      --is_training 1 --root_path "$(dirname -- "$train")" --data_path "$(basename -- "$train")" \
      --val_path "$validation" --test_path "$test" --model_id "${dataset}_${model}_run${index}" \
      --model "$model" --data custom --features M --seq_len "${SEQ_LEN:-200}" --pred_len 20 \
      --enc_in "$channels" --c_out "$channels" --num_class "$classes" --period_len 24 --model_type mlp \
      --epsilon "$epsilon" --loss cross_entropy --train_epochs "${EPOCHS:-10}" --patience "${PATIENCE:-1}" \
      --itr 1 --batch_size "${BATCH_SIZE:-50}" --learning_rate "${LR:-0.05}" --lradj type1 \
      --gpu "$gpu" --use_gpu "$use_gpu" --num_workers "${NUM_WORKERS:-0}" --checkpoints "$checkpoint"
  done
)

launch_main() {
  set -euo pipefail
  local task="${1:---help}"
  case "$task" in
    -h|--help) usage; return ;;
    classification|forecasting|anomaly_detection|concept_drift|structured_classification|structured_forecasting|spas_prepare|spas_classification|all) ;;
    *) usage >&2; fail "Unknown task: $task" ;;
  esac
  shift
  parse_launch_args "$@"
  configure_paths
  configure_task_paths "$task"
  if (( SHOW_PATHS )); then show_paths; return; fi
  case "$task" in
    structured_*|spas_*) (( ! RESUME )) || fail "--resume is not supported for $task; choose a new output directory." ;;
  esac
  case "${DEVICE:-auto}" in auto|cpu) ;; cuda:[0-9]*) [[ "${DEVICE#cuda:}" =~ ^[0-9]+$ ]] || fail "Invalid DEVICE" ;; *) fail "Invalid DEVICE: ${DEVICE}" ;; esac
  init_python
  if [[ "$task" == all ]]; then
    local next
    # Each task runs in a fresh process so its settings cannot leak to the next.
    for next in classification forecasting anomaly_detection concept_drift; do
      bash "$PROJECT_ROOT/scripts/launch_common.sh" "$next" "$@"
    done
  else
    "launch_$task"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  launch_main "$@"
fi
