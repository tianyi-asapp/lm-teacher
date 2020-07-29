ROOT="save"
VERSION="v3"
TRAIN_FILE=data/train_data.pkl 
size=1024
seed=1

EXP=gpt2-${size}-${method}-${VERSION}-s${seed}
SAVE="${ROOT}/${EXP}"
mkdir -p $SAVE
python -m torch.distributed.launch \
    --nproc_per_node 8 train_dialogue.py \
    --output_dir=$SAVE \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TRAIN_FILE \
    --train_task agent \
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
