import numpy as np
import struct
import requests
from tqdm import tqdm

HEADERS_INFO = {
    "gpt-2": {
        "magic": 20250211,
        "version": 1,
        "token_dtype": np.uint16
    }
}

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def write_datafile(filename, token_ids, model_desc="gpt-2"):
    """
    Saves token data as a .bin file
    - First comes a header with 256 int32
    - Then token ids follows as uint16 for gpt-2
    """
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(token_ids) # number of tokens after 256*4 bytes
    # number of bytes to write
    token_ids = np.array(token_ids, dtype=info["token_dtype"])
    num_bytes = len(header)*4 + len(token_ids) * token_ids.itemsize
    print(f"Writing {len(token_ids)} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(token_ids.tobytes())

    
def write_tokenizer(enc, filename):
    '''
    dump the decoded tokens as bytes in the file , decoded means id -> string, so string part as byte
    '''
    n = enc.max_token_value + 1
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20250214 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token

    with open(filename, "wb") as f:
        # write header to the file
        f.write(header.tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            f.write(struct.pack('>B', length)) # # Write the length as a 1-byte unsigned integer
            f.write(b) # Write the actual bytes
    
    print(f"wrote {filename}")
        

    