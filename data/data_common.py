import numpy as np
import struct

HEADERS_INFO = {
    "gpt-2": {
        "magic": 20250211,
        "version": 1,
        "token_dtype": np.uint16
    }
}

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

    with open(filename, "rb") as f:
        # write header to the file
        f.write(header.tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert b < 256, f"Token length exceeds 255: {length}"
            f.write(struct.pack('>B', length)) # # Write the length as a 1-byte unsigned integer
            f.write(b) # Write the actual bytes
    
    print(f"wrote {filename}")
        

    