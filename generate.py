from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import trange
import logging
import argparse
import os

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="../sales_save/v2/gpt2-1024--v2-s1-medium")
parser.add_argument("--data_path", type=str, default="./sales_data/val_data_v2_context_ref_pair.pkl")
parser.add_argument("--output_dir", type=str, default="./sales_out")
parser.add_argument("--p", type=float)
parser.add_argument("--ngram_block", type=int, default=0)
args = parser.parse_args()


tokenizer = GPT2Tokenizer.from_pretrained(args.model_path, pad_token="__PAD__")
model = GPT2LMHeadModel.from_pretrained(args.model_path)
model.cuda()
# IMPORTANT: Note that setting the <PAD> token like this itn the constructor gives the
# pad_token the pad_token_id = 50256, which normally belongs to <BOS> token_ids in GPT2
# This is a very ugly way that works at the moment of setting the pad_token_id to the <BOS> token that is already included in the vocab size. This will be updated in the coming weeks! # noqa: E501

context_ref_pairs = torch.load(args.data_path)

prompt_text = []
reference_text = []
for pair in context_ref_pairs[:5000]:
    prompt_text.append(" ".join(pair[0]))
    reference_text.append(pair[1].strip())


def generate_batch(prompt_batch, num_tokens_to_produce=50):
    #    encode plus batch handles multiple batches and automatically creates attention_masks
    tokens = [tokenizer.encode(l) for l in prompt_batch]
    max_len = 1024 - num_tokens_to_produce
    for i, t in enumerate(tokens):
        if len(t) > max_len:
            tokens[i] = t[-max_len:]
    prompt_batch = [tokenizer.decode(t) for t in tokens]
    seq_lens = [len(t) for t in tokens]
    seq_len = max(seq_lens)
    assert seq_len <= max_len

    encodings_dict = tokenizer.batch_encode_plus(
        prompt_batch, max_length=seq_len, pad_to_max_length=True, add_prefix_space=True
    )

    # ideally we should be able to just input the following two variables to the function model.generate() ... => to be implemented soon!  # noqa: E501
    input_ids = torch.tensor(encodings_dict["input_ids"]).to(model.device)
    attn_mask = torch.tensor(encodings_dict["attention_mask"]).to(model.device)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    eos_not_in_sents = torch.ones(input_ids.shape[0]).long().to(model.device)

    # we need to get the token ids of the last non-padded value
    last_non_masked_idx = (torch.sum(attn_mask, dim=1) - 1).to(model.device)
    start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, len(tokenizer)).unsqueeze(1).to(model.device)
    past = None

    # get correct position ids
    position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])]).to(model.device)
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i] :] = position_ids_slice[last_non_masked_idx[i]]

    generated_output = []
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        min_length=input_ids.size(1) + 2,
        max_length=num_tokens_to_produce + input_ids.size(1),
        top_p=args.p,
        do_sample=True,
        no_repeat_ngram_size=args.ngram_block,
        pad_token_id=tokenizer.pad_token_id,
        bad_words_ids=[
            tokenizer.encode("__PAD__"),
            tokenizer.encode("__AGENT__"),
            tokenizer.encode("__CUSTOMER__"),
            tokenizer.encode("__END_OF_TURN__"),
        ],
    )

    return output_sequences[:, input_ids.size(1) :]


batch_size = 1
generated_outputs = []
for batch_start in trange(0, len(prompt_text), batch_size):
    generated_output = generate_batch(prompt_text[batch_start : batch_start + batch_size])
    for i in range(generated_output.size(0)):
        generated_outputs.append(generated_output[i])

os.makedirs(args.output_dir, exist_ok=True)
output_path = f"{args.output_dir}/hypothesis_p{args.p}_ngramblock{args.ngram_block}.txt"
generated_text = []
with open(output_path, "w") as f:
    for i in range(len(generated_outputs)):
        text = tokenizer.decode(generated_outputs[i], skip_special_tokens=False)
        f.write(text + "\n")
        generated_text.append(text)

for i in range(len(prompt_text)):
    print(f"History: {prompt_text[i]}")
    print(f"Real Customer: {reference_text[i]}")
    print(f"Generated Customer: {generated_text[i]}")
    print("=" * 120)

with open(f"{args.output_dir}/hypothesis_p{args.p}_ngramblock{args.ngram_block}.html", "w") as f:
    for pair, gen in zip(context_ref_pairs, generated_text):
        for i in range(0, len(pair[0]) - 1, 2):
            if pair[0][i] == "__CUSTOMER__":
                tag = '<p style="color:blue">'
            else:
                tag = '<p style="color:red">'
            f.write(f"{tag} {pair[0][i]} {pair[0][i + 1]} <p>")
            f.write("\n")
        f.write(f'<p style="color:green"> Bot: {gen} <p>')
        f.write(f'<p style="color:DarkOrange"> Gold: {pair[1]} <p>\n')
        f.write("<br> <br>")
