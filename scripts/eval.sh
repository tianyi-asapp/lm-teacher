ROOT="sales_save"
VERSION="v2"
TRAIN_FILE=sales_data/train_data_v2.pkl 
VAL_FILE=sales_data/val_data_v2.pkl 
size=1024
seed=1

EXP=gpt2-${size}-${method}-${VERSION}-s${seed}
SAVE="${ROOT}/v2/${EXP}-medium"

python train_dialogue.py \
    --output_dir=$SAVE \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_all_checkpoints \
    --eval_data_file=$VAL_FILE \
    --train_task customer \
    --save_steps 10000 \
    --logging_steps 1000 \
    --evaluate_during_training \
    --num_train_epochs 5 \
    --per_gpu_train_batch_size 1 \
    --block_size $size \
    --fp16_opt_level O2 \
    --seed $seed \
    --block_method none \
    --overwrite_output_dir \
    --fp16
