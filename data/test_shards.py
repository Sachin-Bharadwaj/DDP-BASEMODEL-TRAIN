import tiktoken
import struct

tokenizer = tiktoken.get_encoding('gpt2')
# load one of the saved shard just to check
file = "data/edu_fineweb10B/edu_fineweb_val_000000.bin"
with open(file, 'rb') as f:
    header_bytes = f.read(256*4)
    byte_data = f.read()
    # Calculate the number of uint16 values
    num_uint16 = len(byte_data) // 2
    
    # Unpack the bytes into uint16 integers
    uint16_values = struct.unpack(f'{num_uint16}H', byte_data)
        

print(tokenizer.decode(uint16_values))