# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import math
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from models import GPT2LMHeadModelOneHotTag


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-onehot": (GPT2Config, GPT2LMHeadModelOneHotTag, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
}
IGNORE_INDEX = -100


def construct_input(example, tokenizer, block_method="none", end_of_turn=False):
    # different input format -> commented out
    # text = f"__CUSTOMER__ {example['first_utt']}"
    # text = tokenizer.encode(text, add_prefix_space=True)
    # mask_customer = [False] * len(text)
    # mask_agent = [False] * len(text)
    text = []
    mask_customer = []
    mask_agent = []

    # if end_of_turn and example["spkr"][0] == "agent":
    #     text += tokenizer.encode("__END_OF_TURN__")
    #     mask_agent += [mask_agent[-1]]
    #     mask_customer += [mask_customer[-1]]

    if block_method == "block-onehot":
        iccid_span = [-1, 0]
        meid_span = [-1, 0]

    for i, (utt, spkr) in enumerate(zip(example["text"], example["spkr"])):
        if block_method == "block-tag":
            if i == example.get("iccid", (-1, -1))[0]:
                text += tokenizer.encode("__BLOCK_ICCID_START__")
                mask_agent += [False]
                mask_customer += [False]
            if i == example.get("iccid", (-1, -1))[1]:
                text += tokenizer.encode("__BLOCK_ICCID_END__")
                mask_agent += [False]
                mask_customer += [False]
            if i == example.get("meid", (-1, -1))[0]:
                text += tokenizer.encode("__BLOCK_MEID_START__")
                mask_agent += [False]
                mask_customer += [False]
            if i == example.get("meid", (-1, -1))[1]:
                text += tokenizer.encode("__BLOCK_MEID_END__")
                mask_agent += [False]
                mask_customer += [False]
        elif block_method == "block-onehot":
            if i == example.get("iccid", (-1, -1))[0]:
                iccid_span[0] = len(text)
            if i == example.get("iccid", (-1, -1))[1]:
                iccid_span[1] = len(text)
            if i == example.get("meid", (-1, -1))[0]:
                meid_span[0] = len(text)
            if i == example.get("meid", (-1, -1))[1]:
                meid_span[1] = len(text)

        tmp = tokenizer.encode(f" __{spkr.upper()}__ {utt} {tokenizer.eos_token}", add_prefix_space=True)
        text += tmp
        if spkr == "agent":
            mask_agent += [False] + [True] * (len(tmp) - 1)
            mask_customer += [False] * len(tmp)
        else:
            mask_customer += [False] + [True] * (len(tmp) - 1)
            mask_agent += [False] * len(tmp)

        if end_of_turn and ((i + 1 == len(example["spkr"])) or (example["spkr"][i] != example["spkr"][i + 1])):
            text += tokenizer.encode("__END_OF_TURN__")
            mask_agent += [mask_agent[-1]]
            mask_customer += [mask_customer[-1]]

    if block_method == "block-onehot":
        tags = torch.zeros((len(text), 2))
        tags[iccid_span[0] : iccid_span[1], 0] = 1
        tags[meid_span[0] : meid_span[1], 1] = 1
        return text, mask_customer, mask_agent, tags
    else:
        return text, mask_customer, mask_agent


class DialogueDataset(Dataset):
    def __init__(
        self, tokenizer, file_path="train", split="train", block_size=512, block_method="none", end_of_turn=False
    ):
        assert os.path.isfile(file_path)
        assert split in {"train", "val", "debug"}
        directory, filename = os.path.split(file_path)
        prefix = ""
        if end_of_turn:
            prefix += "_eot"
        if block_method != "none":
            prefix += f"_{block_method}"
        cached_features_file = os.path.join(
            directory, "cached_lm{}_{}_{}_{}".format(prefix, block_size, split, filename)
        )

        if os.path.exists(cached_features_file) and split != "debug":
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            raw_data = torch.load(file_path)
            raw_data = list(raw_data.values())
            # n_val = 5000
            # n_train = len(raw_data) - n_val
            # if split == "train":
            #     raw_data = raw_data[:n_train]
            # elif split == "val":
            #     raw_data = raw_data[n_train:]
            # elif split == "debug":
            #     raw_data = raw_data[:20]
            # elif split == "all":
            #     pass
            # else:
            #     raise ValueError(f"split={split}")

            examples = [
                construct_input(example, tokenizer, block_method=block_method, end_of_turn=end_of_turn)
                for example in tqdm(raw_data)
            ]

            self.examples = []
            for example in examples:
                for i in range(0, len(example[0]), block_size):
                    tmp = tuple(x[i : i + block_size] for x in example)

                    n_pad = block_size - len(tmp[0])
                    if n_pad > 0:
                        if block_method == "block-onehot":
                            tmp = (
                                tmp[0] + [tokenizer.pad_token_id] * n_pad,
                                tmp[1] + [False] * n_pad,
                                tmp[2] + [False] * n_pad,
                                torch.cat([tmp[3], tmp[3].new_zeros(n_pad, tmp[3].size(1))], dim=0),
                            )
                        else:
                            tmp = (
                                tmp[0] + [tokenizer.pad_token_id] * n_pad,
                                tmp[1] + [False] * n_pad,
                                tmp[2] + [False] * n_pad,
                            )
                    self.examples.append(tuple(torch.tensor(x) for x in tmp))

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
        # example = self.examples[item]
        # return torch.tensor(example[0]), torch.tensor(example[1]), torch.tensor(example[2])


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = DialogueDataset(
        tokenizer,
        file_path=args.eval_data_file if evaluate else args.train_data_file,
        split="val" if evaluate else "train",
        block_size=args.block_size,
        block_method=args.block_method,
        end_of_turn=args.end_of_turn,
    )
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if args.block_method == "block-onehot":
                tag_vecs = batch[3].to(args.device)
            inputs, mask_customer, mask_agent = batch[:3]
            # inputs, labels = mask_tokens(text, tokenizer, args) if args.mlm else (text, text)
            inputs = inputs.to(args.device)
            with torch.no_grad():
                label_mask = mask_customer.new_zeros(mask_customer.size())
                if "customer" in args.train_task:
                    label_mask += mask_customer
                if "agent" in args.train_task:
                    label_mask += mask_agent
                if args.train_task == "all":
                    label_mask = labels != tokenizer.pad_token_id

                label_mask = label_mask.to(args.device)
                labels = inputs.clone().detach()
                labels[label_mask == 0] = IGNORE_INDEX
            model.train()
            # outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            if args.block_method == "block-onehot":
                outputs = model(inputs, labels=labels, tag_vecs=tag_vecs)
            else:
                outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    log_loss = (tr_loss - logging_loss) / args.logging_steps
                    lr = scheduler.get_lr()[0]
                    tb_writer.add_scalar("lr", lr, global_step)
                    tb_writer.add_scalar("loss", log_loss, global_step)
                    tb_writer.add_scalar("ppl", math.exp(log_loss), global_step)
                    logger.info(f"train step: {global_step} loss: {log_loss} ppl: {math.exp(log_loss)} lr: {lr}")
                    logging_loss = tr_loss
                    # evaluate
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        msg = ""
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            msg += f" {key}: {value}"
                        logger.info(f"test step: {global_step}{msg}")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", eval_output_dir=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if eval_output_dir is None:
        eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            if args.block_method == "block-onehot":
                tag_vecs = batch[3].to(args.device)
            inputs, mask_customer, mask_agent = batch[:3]
            inputs = inputs.to(args.device)
            label_mask = mask_customer.new_zeros(mask_customer.size())
            if "customer" in args.eval_task:
                label_mask += mask_customer
            if "agent" in args.eval_task:
                label_mask += mask_agent
            if args.eval_task == "all":
                label_mask = labels != tokenizer.pad_token_id
            label_mask = label_mask.to(args.device)
            labels = inputs.clone().detach()
            labels[label_mask == 0] = IGNORE_INDEX
            # outputs = model(inputs, masked_lm_labels=inputs) if args.mlm else model(inputs, labels=inputs)
            if args.block_method == "block-onehot":
                outputs = model(inputs, labels=labels, tag_vecs=tag_vecs)
            else:
                outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = math.exp(eval_loss)

    result = {
        "loss": eval_loss,
        "ppl": perplexity,
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--train_task", type=str, default="customer", choices=["all", "agent", "customer", "agent+customer"]
    )
    parser.add_argument("--eval_task", type=str, default=None, choices=["all", "agent", "customer", "agent+customer"])
    parser.add_argument(
        "--block_method",
        type=str,
        default="none",
        choices=["none", "block-tag", "block-onehot"],
        help="how we use the block infomation",
    )
    parser.add_argument("--end_of_turn", action="store_true", help="to predict end of turn tokens")
    args = parser.parse_args()

    if args.eval_task is None:
        args.eval_task = args.train_task
    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    assert ("onehot" in args.model_type) == ("onehot" in args.block_method)

    fh = logging.FileHandler(os.path.join(args.output_dir, "logs.txt"))
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(
        args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config
    )

    # add new tokens
    additional_tokens = [
        "__AGENT__",
        "__CUSTOMER__",
        "__ICCID_NUM__",
        "__MEID_NUM__",
        "__PHONE_NUM__",
        "__PIN_NUM__",
        "<delayed>",
    ]
    if args.block_method == "block-tag":
        additional_tokens += [
            "__BLOCK_ICCID_START__",
            "__BLOCK_ICCID_END__",
            "__BLOCK_MEID_START__",
            "__BLOCK_MEID_END__",
        ]
    if args.end_of_turn:
        additional_tokens += ["__END_OF_TURN__"]
    tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens, "pad_token": "__PAD__"})
    model.resize_token_embeddings(len(tokenizer))
    if args.block_method == "block-onehot":
        model.set_tag_embedding(2)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, eval_output_dir=checkpoint, prefix=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
