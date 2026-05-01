#!/bin/bash
# Launch all epsilon × method combos in separate screen sessions
# Epsilon: 1, 1.5, 2, ..., 6 (11 values) × 3 methods = 33 screens
# Distribute across GPU 4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
R=
RP=
DP=traffic.csv
MID=traffic
DN=custom
MN=PatchTST
SL=80; PL=20

RT=
RS=
RPO=
mkdir -p "$RT" "$RS" "$RPO"

# All three methods now use lr=0.012, clip=1
LR=0.005
CLIP=1

GPUS=(3 4 5 6)
MAX_JOBS_PER_GPU=1
declare -A gpu_jobs

# Initialize job counts
for gpu in "${GPUS[@]}"; do
  gpu_jobs[$gpu]=0
done

EPSILONS=(10 15)

for eps in "${EPSILONS[@]}"; do
  for method in traffic; do
    # Find a GPU with available slots
    assigned_gpu=-1
    while [ $assigned_gpu -eq -1 ]; do
      for gpu in "${GPUS[@]}"; do
        # Count running screens for this specific GPU by checking screen names
        # We look for screens named traffic_PatchTST_gpu${gpu}_...
        current_jobs=$(screen -ls | grep -c "traffic_PatchTST_gpu${gpu}_")
        
        if [ "$current_jobs" -lt "$MAX_JOBS_PER_GPU" ]; then
          assigned_gpu=$gpu
          break
        fi
      done
      
      if [ $assigned_gpu -eq -1 ]; then
        echo "All GPUs are full (max $MAX_JOBS_PER_GPU jobs each). Waiting 30 seconds..."
        sleep 10
      fi
    done

    gpu=$assigned_gpu

    # Include GPU ID in screen name for tracking
    screen_name="traffic_PatchTST_gpu${gpu}_${method}_eps${eps}"

    if [ "$method" = "traffic" ]; then
      des=baseline
      script="${R}/run_Exp.py"
      out="${RT}/traffic_len${SL}_pred${PL}_eps${eps}_lr${LR}_clip${CLIP}.txt"
    # elif [ "$method" = "strat" ]; then
    #   des=strata
    #   script="${R}/run_Exp_stratification.py"
    #   out="${RS}/traffic_strat_len${SL}_pred${PL}_eps${eps}_lr${LR}_clip${CLIP}.txt"
    else
      des=poisson
      script="${R}/run_Exp_spaced_sampling_poisson.py"
      out="${RPO}/traffic_strat_poisson_len${SL}_pred${PL}_eps${eps}_lr${LR}_clip${CLIP}.txt"
    fi

    # # Skip if result already exists with valid data
    # if [ -f "$out" ]; then
    #   val=$(tail -1 "$out" | awk '{print $NF}')
    #   if [[ "$val" =~ ^[0-9]+\.[0-9]+$ ]]; then
    #     echo "[SKIP] $screen_name : result exists ($val)"
    #     gpu_idx=$((gpu_idx - 1))  # don't consume a GPU slot
    #     continue
    #   fi
    # fi

    cmd="cd ${SCRIPT_DIR} && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate mts_env && \
echo -e 'epsilon\tmean_test_loss' > '${out}' && \
python -u ${script} \
  --is_training 1 --root_path ${RP} --data_path ${DP} \
  --model_id ${MID}_${SL}_${PL} --model ${MN} --data ${DN} \
  --des ${des} --features M --seq_len ${SL} --pred_len ${PL} \
  --enc_in 371 --train_epochs 100 --patience 5 --micro_batch_size 25\
  --itr 30 --batch_size 50 --learning_rate ${LR} \
  --dp_sigma 5 --dp_delta 1e-5 --privacy_budget_limit ${eps} --w 0.005 \
  --devices ${gpu} --c_out 371 --clipping_norm ${CLIP} \
  --result_file '${out}' && \
echo 'DONE: ${screen_name}'; \
exec bash"

    screen -dmS "$screen_name" bash -c "$cmd"
    echo "[LAUNCH] $screen_name on GPU $gpu"
    sleep 1  # small delay to avoid race conditions
  done
done

echo ""
echo "Total screens launched:"
screen -ls | grep -cE "traffic_eps|strat_eps|poisson_eps"
echo ""
echo "All screens:"
screen -ls | grep -E "traffic_eps|strat_eps|poisson_eps" | sort
