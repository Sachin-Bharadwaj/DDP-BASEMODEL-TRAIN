import os, sys

# for some reason, the root directory of this project is not being added to the python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from config.cfg import GPT2Config
from model.GPT2 import GPTModel
import tiktoken
from dataclasses import asdict
from data.hellaswag import sanitize_state_dict
import torch

enc = tiktoken.get_encoding("gpt2")
# --------------------------------------------

if __name__ == "__main__":
    start_monologue = " As I went to Mars, I saw"

    start_monologue_enc = [enc.eot_token] + enc.encode_ordinary(start_monologue)

    # define the model with its config
    gptcfg = asdict(GPT2Config())
    model = GPTModel(gptcfg)
    device = "cpu"

    # get the final model checkpoint
    chkpoint_file = "/Users/sachinbharadwaj/Sachin/BuildingLLMs/outdir/finewebedu-10BT/chkpoint_38000.pth"
    # sanitize state dict
    new_model_state_dict = sanitize_state_dict(chkpoint_file)
    # load the state_dict
    model.load_state_dict(new_model_state_dict)
    model.to(device)

    model.eval()
    # we will kickoff the generation with "<|endoftext|>", which designates the start of a new seqeunce
    start_id = start_monologue_enc
    xg = torch.tensor(start_id, dtype=torch.long, device=device)[None, ...]
    max_new_tokens = 256
    temperature = 1.0
    top_k = 40
    yg = model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
    print("---------------------------------")
    print(enc.decode(yg[0].tolist()))
    print("---------------------------------")