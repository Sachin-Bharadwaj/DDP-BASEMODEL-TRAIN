import tiktoken
import os
import requests
from tqdm import tqdm
from data_common import write_datafile


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespare")

def download_file(url, output_path):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Raise an error for bad responses
    response.raise_for_status()
    
    # Get the total file size from the headers
    total_size = int(response.headers.get('content-length', 0))
    
    # Open the output file in binary write mode
    with open(output_path, 'wb') as file:
        # Initialize the tqdm progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path, ascii=True) as pbar:
            # Iterate over the response data in chunks
            for data in response.iter_content(chunk_size=1024):
                # Write the data to the file
                file.write(data)
                # Update the progress bar
                pbar.update(len(data))

def tokenize(filename, tokenizer):
    # lets split the tinyshakespare into section based on "\n\n" and treat every section as new document
    with open(filename, "r") as f:
        data = f.read()
    sections = data.split("\n\n")
    eot_token = tokenizer.eot_token
    token_ids = []
    for text in sections:
        token_ids.append(eot_token)
        token_ids_ = tokenizer.encode(text)
        token_ids.extend(token_ids_)
    
    # lets take first 32768 as val tokens and remaining as train tokens
    val_tokens = token_ids[0:32768]
    train_tokens = token_ids[32768:]
    # save these to a file
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens, model_desc="gpt-2")
    write_datafile(train_filename, train_tokens, model_desc="gpt-2")
    


if __name__ == "__main__":
    # download the tiny shakespare
    
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokenize(data_filename, tokenizer)
    
