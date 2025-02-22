import os, sys
import requests
import json
import pdb
import glob

import torch 
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


from dataclasses import asdict
from collections import OrderedDict

# for some reason, the root directory of this project is not being added to the python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from data.data_common import download_file
from config.cfg import GPT2Config
from model.GPT2 import GPTModel

#  ----------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    '''
    split: "train" or "val" or "test"
    '''
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def render_example(example):
    '''
    Given the example as a dict, render it as 3 torch tensors:
    - tokens : torch tensor of size num_example x maxlen
    - mask : torch tensor of size num_example x maxlen, mask=0 for context and 1 otherwise
    - label (the index of the correct completion, which we hope has the highest likelihood)
    '''
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    token_rows = [] # will hold (ctx + end) tokens for every example, since each example has 4 endings, this will be [[], [], [], []]
    mask_rows = [] # ctx mask will be 0, end mask be will 1, [[], [], [], []] , 4 endings for every example
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepend " " coz GPT2 tokenizer
        # concatenate ctx and end
        token_row = ctx_tokens + end_tokens
        token_rows.append(token_row)
        # mask is 0 for tokens in ctx, 1 for tokens in end
        mask = [0]*len(ctx_tokens) + [1]*len(end_tokens)
        mask_rows.append(mask)

    # before comverting to tensors, have to be careful during collation coz number of tokens in each row can differ due to ending of unequal length for every example
    max_len = max([len(row) for row in token_rows])
    tokens = torch.zeros((len(token_rows), max_len), dtype=torch.long) # NOTE: beyond context+ending length, the token ids are appended with zeros while collating
    mask = torch.zeros((len(mask_rows), max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(token_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
    return tokens, mask, torch.tensor(label, dtype=torch.long)
    


def iterate_examples(split):
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            yield json.loads(line)

@torch.no_grad()
def evaluate(model, model_state_dict, device, print_every=500):
    torch.set_float32_matmul_precision('high') # use tf32

    # load the pre-trained weights
    print(f"loading state dict ...")
    model.load_state_dict(model_state_dict)
    model.to(device)

    num_total = 0
    num_correct_norm = 0
    num_correct_unnorm = 0
    # iterate over examples
    for example in iterate_examples("val"):
        tokens, mask, label = render_example(example)
        bs = tokens.shape[0]
        tokens = tokens.to(device)
        mask = mask.to(device)
        label = label.to(device)

        # forward pass, target=-1 only returns all the logits for all elements in the sequence, no loss is computed or returned
        logits, _ = model(tokens, targets=-1, return_logits=True) # logits: [num_endings,max_len,vocab_size]
        # now we have both predictions = logits, ground truth = input (tokens ids)
        # so we can compute the auto-regressive loss at every position in the sequence
        preds = logits[...,:-1,:]
        gt = tokens[...,1:]
        # flatten the preds and gt
        preds = preds.flatten(0,1) #[bs*max_seq_len-1, vocab_size]
        gt = gt.flatten() #[bs*max_seq_len-1]
        # compute the loss at every position in sequence
        
        loss = F.cross_entropy(preds, gt, reduction='none') # #[bs*max_seq_len-1] reduction = 'none' no reduction applied
        loss = loss.view(bs,-1) #[bs, max_seq_len-1]
        # now get the loss for just completion part by applying the mask
        # first mask needs to shifted by 1
        mask_shifted = mask[...,1:]
        mask_loss = loss * mask_shifted #[bs, max_seq_len-1]
        # sum and divide by # of 1s in the mask
        sum_loss = mask_loss.sum(dim=1)
        avg_loss = sum_loss / mask_shifted.sum(dim=1)
        # now we have loss for every sample in the batch (i.e 4 endings per example)
        # one with the lowest loss should be the most likely completion
        pred_unnormalized = sum_loss.argmin().item() # un-normalized w.r.t ending completion length
        pred_normalized = avg_loss.argmin().item()  # normalized w.r.t ending completion length
        
        # accumulate stats
        num_total +=1
        num_correct_norm += int(pred_normalized == label.item())
        num_correct_unnorm += int(pred_unnormalized == label.item())
        print(f"total examples: {num_total},  unnorm-acc:{num_correct_unnorm/num_total : .4f} norm-acc:{num_correct_norm}/{num_total} = {num_correct_norm/num_total : .4f}")

        if num_total < 10:
            
            print("-------------------")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item()}) : {end}")
            print(f"predicted: {pred_normalized}, actual: {label.item()}")  

    return (num_total, num_correct_norm, num_correct_unnorm)

        
def sanitize_state_dict(chkpoint_file, device='cpu'):
    '''
    this functions strips the extra string appended by DDP wrapper while saving the checkpoint
    after running this script, the checkpoint can be loaded w/o wrapped your model inside DDP
    '''

    new_model_state_dict = OrderedDict()
    state_dict = torch.load(chkpoint_file, map_location=torch.device('cpu'))
    model_state_dict = state_dict["model_state_dict"]
    for k, v in model_state_dict.items():
        new_key = k.split(".")[1:]
        new_key= ".".join(new_key)
        new_model_state_dict[new_key] = v
    return new_model_state_dict


if __name__ == "__main__":
    # download the hellaswag spits
    download("test")
    download("val")

    # print one example from "val" set
    print(iterate_examples("val").__next__())

    # compute eval on hellaswag "val" split
    chkpoint_files = glob.glob("/Users/sachinbharadwaj/Sachin/BuildingLLMs/outdir/finewebedu-10BT/*.pth")
    stats_file = os.path.join(DATA_CACHE_DIR, "hellaswag-val-stats.json" )
    final_results = {}
    
    for file in chkpoint_files:
        key = file.split("/")[-1].split(".")[0].split("_")[-1] # iter number of chkpoint file
        # sanitize state dict
        new_model_state_dict = sanitize_state_dict(file)
        # define the model architecture
        gptcfg = asdict(GPT2Config())
        model = GPTModel(gptcfg)
        result = evaluate(model, new_model_state_dict, device="cpu")
        final_results[key] = result

    with open(stats_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    