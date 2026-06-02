if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Limit OpenMP/MKL threads to reduce CPU usage
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_MAX_THREADS=2

model_name=

root_path_name=path/noised_data_original/epsilon_1.0
val_path_name=path/noised_data/val_data.csv
test_path_name=path/noised_data/test_data.csv
model_id_name=DynamicMTS
data_name=custom

seq_len=80
pred_len=20


python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path train_noised_0.csv \
    --val_path $val_path_name \
    --test_path $test_path_name \
    --model_id ${model_id_name}_run0_${seq_len}_${pred_len}_single \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 17 \
    --period_len 24 \
    --model_type 'mlp' \
    --train_epochs 10 \
    --patience 1 \
    --itr 1 --batch_size 50 --learning_rate 0.05 \
    --lradj type1 \
    --c_out 17 \
    --gpu 4 \
    --num_workers 0 \
       