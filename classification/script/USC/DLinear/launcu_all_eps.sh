SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
R=
RP=
TRAIN_PATH=""
TEST_PATH=""
RESULT_PATH=""
mkdir -p ${RESULT_PATH}

MODEL="DLinear_Stable"

SEQ_LEN=200
NUM_KERNELS=10
ENC_IN=6
NUM_CLASSES=12
BATCH_SIZE=50
NUM_WORKERS=10
EPOCHS=30
ITR=30
W=0.005

LR=0.1
LR_STEP_SIZE=20
LR_GAMMA=0.9
DP_SIGMA=5
CLIPPING_NORM=1

LABEL_MODE="sequence"

GPUS=(0 4 5 7)
gpu_idx=0

EPSILONS=(1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6)

for eps in "${EPSILONS[@]}"; do
  for method in baseline poisson; do
    gpu=${GPUS[$((gpu_idx % 4))]}
    gpu_idx=$((gpu_idx + 1))

    screen_name="USC_DLinear_LR0.1_${method}_eps${eps}"

    if [ "$method" = "baseline" ]; then
      des=baseline
      MODEL_ID="baseline_DLinear"
      script="${R}/run_DPSGD_TSC.py"
      out="${RESULT_PATH}/baseline_${MODEL}_len${SEQ_LEN}_eps${eps}_lr${LR}_clip${CLIPPING_NORM}_dps${DP_SIGMA}_bz${BATCH_SIZE}.txt"
    else
      des=poisson
      MODEL_ID="poisson_DLinear"
      script="${R}/run_poisson_TSC.py"
      out="${RESULT_PATH}/poisson_${MODEL}_len${SEQ_LEN}_eps${eps}_lr${LR}_clip${CLIPPING_NORM}_dps${DP_SIGMA}_bz${BATCH_SIZE}.txt"
    fi

    cmd="cd ${SCRIPT_DIR} && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate mts_env && \
echo -e 'epsilon\tmean_test_loss' > '${out}' && \
python -u ${script} \
  --is_training 1 \
        --model_id ${MODEL_ID} \
        --model ${MODEL} \
        --seq_len ${SEQ_LEN} \
        --num_kernels ${NUM_KERNELS} \
        --enc_in ${ENC_IN} \
        --num_classes ${NUM_CLASSES} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --train_epochs ${EPOCHS} \
        --itr ${ITR} \
        --learning_rate ${LR} \
        --gpu ${gpu} \
        --dp_sigma ${DP_SIGMA} \
        --clipping_norm ${CLIPPING_NORM} \
        --privacy_budget_limit ${eps} \
        --label_mode ${LABEL_MODE} \
        --w ${W} \
        --train_path ${TRAIN_PATH} \
        --test_path ${TEST_PATH} \
        --result_file ${out} \
        --checkpoints ${RESULT_PATH}/checkpoints/ \
        --lr_step_size ${LR_STEP_SIZE} \
        --lr_gamma ${LR_GAMMA}
echo 'DONE: ${screen_name}'; \
exec bash"

    screen -dmS "$screen_name" bash -c "$cmd"
    echo "[LAUNCH] $screen_name on GPU $gpu"
    sleep 2  # small delay to avoid race conditions
  done
done

# echo ""
# echo "Total screens launched:"
# screen -ls | grep -cE "baseline_eps|poisson_eps"
# echo ""
# echo "All screens:"
# screen -ls | grep -E "baseline_eps|poisson_eps" | sort
