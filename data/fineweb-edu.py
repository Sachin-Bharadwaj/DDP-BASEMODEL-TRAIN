# FineWeb-edu has sample-350BT, sample-100BT, sample-10BT
# sample-350BT was sampled randomly from entire dataset
# sample-100BT was sampled from sample-350BT
# sample-10BT was sampled from sample-100BT

from datasets import load_dataset
import tiktoken
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import set_start_method
from functools import partial
from tqdm import tqdm
from data_common import write_datafile
import os
import pdb


def tokenize(doc: dict, tokenizer):
    '''
    Tokenize the input doc using gpt2 tokenizer
    1. doc: dict object , doc['text'] is the text we are interested in
    2. tokenizer: tokenizer object for gpt2 using tiktoken

    Outputs:
    1. token_ids: np.array of uint16
    '''
    text = doc['text']
    token_ids = []
    token_ids_ = tokenizer.encode_ordinary(text) # returns list of ints
    # get "<|endoftext|>" token id
    eot_id = tokenizer.eot_token
    token_ids.append(eot_id) # putting eot token at the begining to signify the start of new text sequence.
    token_ids.extend(token_ids_)
    # convert the token_ids to np.array and check that all tokens can be 
    # represented as uint16
    token_ids_np = np.array(token_ids)
    assert (0 <= token_ids_np).all() and (token_ids_np < 2**16).all(), "vocab size is too large for uint16"
    token_ids_np = token_ids_np.astype(np.uint16)
    return token_ids



# create dataparallel tokenization using mp.pool
if __name__ == "__main__":

    tokenizer_gpt2 = tiktoken.get_encoding('gpt2')
    vocab_size_tiktoken_gpt2 = tokenizer_gpt2.max_token_value
    num_unsigned_bits_req = np.ceil(np.log2(vocab_size_tiktoken_gpt2))
    print(f"number of unsigned bits required by gpt2 tokenizer: {num_unsigned_bits_req}")

    ds_sample_10BT = load_dataset("HuggingFaceFW/fineweb-edu",
                                name="sample-10BT",  
                                streaming=True)
    # print infomation about the dataset
    print(ds_sample_10BT)

    # this is iterable object with key='train', so access iterable object using 
    # ds_sample_10BT['train'], text can be accessed as ds_sample_10BT['train']['text']
    dummy_sample = []
    max_cnt = 0
    for sample in ds_sample_10BT['train']:
        sample_dict = sample
        dummy_sample.append(sample_dict)
        max_cnt +=1
        if max_cnt > 100:
            break

    print(dummy_sample)
    with open("dummy_sample.txt", 'w', encoding='utf-8') as f:
        for data in dummy_sample:
            text = data['text']
            f.write(text)
            f.write('<|endoftext|>')
        
   
    num_cpu = os.cpu_count()
    num_procs = max(1, num_cpu - 2)
    print(f"number of proc: {num_procs}")
    #set_start_method('fork') # protect the entry point for spawning child processes 
    ds_sample_10BT_train = ds_sample_10BT['train']
    # {{ sharding params
    shard_size = int(100*1e6) # tokens per shard
    token_dtype = np.uint16
    local_dir = "edu_fineweb10B"
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    if not os.path.exists(DATA_CACHE_DIR):
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    dset_name = "edu_fineweb"
    #}}
    with Pool(num_procs) as pool: # set mp context with num processes
        shard_idx = 0
        total_token_cnt = 0
        # pre-allocate buffer to hold current shard
        token_np = np.empty((shard_size,), dtype=token_dtype)
        progress_bar = None
        # ds_sample_10BT['train']
        for token_ids_list in pool.imap(partial(tokenize, tokenizer=tokenizer_gpt2), iter(ds_sample_10BT['train']), chunksize=num_procs):
            #print(token_ids_list, flush=True)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc=f"shard-{shard_idx}", unit="tokens")
            # fill the tokens in the current shard
            if total_token_cnt + len(token_ids_list) < shard_size:
                token_np[total_token_cnt: total_token_cnt + len(token_ids_list)] = token_ids_list
                total_token_cnt += len(token_ids_list)
                # update progress bar
                progress_bar.update(len(token_ids_list))
            else:
                # write the current shard and start a new one
                split = "val" if shard_idx == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{dset_name}_{split}_{shard_idx:06d}.bin")
                # split the data in whatever fits in the current shard, the remainder goes to the next shard
                remainder = shard_size - total_token_cnt
                progress_bar.update(remainder)
                token_np[total_token_cnt:shard_size] = token_ids_list[:remainder]
                # make a write call to dump the shard
                write_datafile(filename, token_np, model_desc="gpt-2")
                shard_idx +=1
                progress_bar = None
                # populate the tokens_np with the remainder from the current shard
                try:
                    token_np[0:len(token_ids_list) - remainder] = token_ids_list[remainder:]
                except Exception as e:
                    pdb.set_trace()
                total_token_cnt = len(token_ids_list) - remainder