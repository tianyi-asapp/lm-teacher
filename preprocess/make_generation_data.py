import re
import os
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import argparse
from collections import defaultdict
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="/persist/projects/gen_auto_suggest/experiment/sales_data/val_data_v2.pkl",
    help="path to the raw csv data file",
)
parser.add_argument(
    "--output_path", type=str, default="sales_data", help="folder to put the output",
)
args = parser.parse_args()


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

all_convos = torch.load(args.data_path)

context_ref_pair = []
for convo_data in tqdm(all_convos.values()):
    context = []
    agent_start = False
    for utt, spkr in zip(convo_data["text"], convo_data["spkr"]):
        agent_start = agent_start or (spkr == "agent")
        context.append(f"__{spkr.upper()}__")
        utt = f"{utt} {tokenizer.eos_token}"
        if agent_start and (spkr == "customer"):
            context_ref_pair.append((tuple(context), utt))
        context.append(utt)

os.makedirs(args.output_path, exist_ok=True)
torch.save(context_ref_pair, os.path.join(args.output_path, "val_data_v2_context_ref_pair.pkl"))
