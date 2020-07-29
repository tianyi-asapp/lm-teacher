# Sales Simulator

This repo contains the code to finetunes a GPT-2 model on ASAPP sales dialogue data.

## Data

We expect a csv file that contains conversations. The columns should have `issue_id` (`conversation_id`),
`utterance` (text), `sender_type` (bot, customer, or agent). The rows are expected to be in the order of the dialogue.

## Preprocessing
```
preprocess
├── make_generation_data.py # this makes the data for generate.py
└── make_train_val.py # this makes the data for train_dialogue.py
```

`python *.py --help` to see the arguments.

## Training
`scripts/sales_customer.sh` contains an example command to run and the hypers I used.
This script is supposed to be run on a 8-V100 machiens (on aws, p3.16xlarge).

When you are not launching the script on a multi-gpu machine (say, you are debugging), replace `python -m torch.distributed.launch --nproc_per_node 8 train_dialogue.py` with `python train_dialogue.py`.

The training log will be saved in `sales_save/v2/gpt2-1024--v2-s1-medium/logs.txt`. `sales_save/v2/gpt2-1024--v2-s1-medium/tb_logs` contains the tensorboard output, which mainly contains the training perplexity.

## Evaluation (Perplexity)
During training, `train_dialogue.py` will not run any validation because distributed training complicates a lot of things. So all checkpoints will be saved to `sales_save/checkpoint-*`. `eval.sh` provides an example command for how to evaluate these saved checkpoints. `eval.sh` will produce `eval_results.txt` in each of the checkpoint folder, which records the evaluation result. Do not evaluate too frequently, otherwise the checkpoints will quickly take up all the disk space. Calculate how many checkpoints you can afford to store before launching the training command.

## Generation
After evaluating each of the checkpoint, select the best checkpoint to simulate a customer. `generate.sh` provides the example command to do this. This script produces a html file under `sales_save`.