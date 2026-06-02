#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PYTHON_ENV="mts_env"
CONDA_PATH=$(conda info --base)
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${PYTHON_ENV}"
PYTHON="python"

FILE_PATHS=(
  "/data/yulian/classifier_dataset/training_data.csv", "/data/yulian/USC_dataset/training_data.csv"
)

# Label perturbation consumes half of the privacy budget.
EPSILON_LIST=(3 2.75 2.5 2.25 2 1.75 1.5 1.25 1 0.75 0.5)

TOTAL=30
BATCH_SIZE=5

# Concurrency limit: allow at most MAX_RUNNING spas_* screen sessions.
LIMIT_PREFIX="spas_"
MAX_RUNNING=30
POLL_SEC=10

count_running_prefix() {
  # The first column of screen -ls is usually PID.session_name.
  # Match only that column to avoid parsing failures across locales or formats.
  screen -ls 2>/dev/null | awk -v p="$LIMIT_PREFIX" '
    $1 ~ "^[0-9]+\\."p {c++}
    END {print c+0}
  '
}

wait_for_slot() {
  local n
  while true; do
    n="$(count_running_prefix)"
    if (( n < MAX_RUNNING )); then
      return 0
    fi
    echo "[throttle] ${LIMIT_PREFIX} running=${n} >= ${MAX_RUNNING}, sleep ${POLL_SEC}s"
    sleep "${POLL_SEC}"
  done
}

count=0
for file_path in "${FILE_PATHS[@]}"; do
  dataset="$(basename "$(dirname "$file_path")")"   # traffic

  for epsilon in "${EPSILON_LIST[@]}"; do
    start=0
    while (( start < TOTAL )); do
      end=$((start + BATCH_SIZE))
      (( end > TOTAL )) && end=$TOTAL

      eps_tag="${epsilon//./_}"
      session_name="spas_${dataset}_eps${eps_tag}_${start}_$((end-1))"

      # Throttle all spas_* sessions.
      wait_for_slot

      echo "Starting: ${session_name} (epsilon=${epsilon}, i=[${start},${end}))"

      screen -dmS "${session_name}" bash -lc "
        cd '${SCRIPT_DIR}' && \
        ${PYTHON} process_data.py \
          --file_path '${file_path}' \
          --epsilon ${epsilon} \
          --start_i ${start} \
          --end_i ${end}
      "

      # Give screen time to write the session into the socket list.
      sleep 1

      count=$((count + 1))
      start=$end
    done
  done
done

echo "All screen sessions have started, total: ${count}."
echo "Run screen -ls to inspect them."
