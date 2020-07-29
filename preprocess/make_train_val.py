import os
import re
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import argparse
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="~/tianyi_champ_sales/champ_sales_2019_2020.csv",
    help="path to the raw csv data file",
)
parser.add_argument(
    "--output_path", type=str, default="../sales_data", help="folder to put the output",
)
args = parser.parse_args()

CHUNKSIZE = 10 ** 6  # amount of lines to load at a time

processed_data = defaultdict(dict)

for i, chunk in enumerate(pd.read_csv(args.data_path, chunksize=CHUNKSIZE)):

    for index, row in tqdm(chunk.iterrows(), total=chunk.shape[0]):
        if row["sender_type"] == "bot":
            continue
        convo_dict = processed_data[row["issue_id"]]
        text = row["utterance"]

        # sometimes the message is empty
        if type(text) != str:
            continue
        # some basic preprocessing. could be improved.
        text = text.replace("\n", "")
        text = re.sub("\{customer_delayed.*\}", "<delayed>", text)
        if "text" not in convo_dict:
            convo_dict["text"] = []
        if "spkr" not in convo_dict:
            convo_dict["spkr"] = []
        convo_dict["text"].append(text)
        convo_dict["spkr"].append("customer" if row["sender_type"] == "customer" else "agent")

convo_ids = np.array(list(processed_data.keys()))

# random sample some conversations to be validation
val_ids = np.random.choice(convo_ids, 5000, replace=False)
train_ids = np.array(list(set(convo_ids) - set(val_ids)))

train_data = {idx: processed_data[idx] for idx in train_ids}
val_data = {idx: processed_data[idx] for idx in val_ids}

os.makedirs(args.output_path, exist_ok=True)
torch.save(train_data, os.path.join(args.output_path, "train_data_v2.pkl"))
torch.save(val_data, os.path.join(args.output_path, "val_data_v2.pkl"))
