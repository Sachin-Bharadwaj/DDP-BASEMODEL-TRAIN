import torch
import torch.nn as nn
from dataclasses import asdict
import numpy as np
import glob
import math
import wandb


import os, sys
import inspect

# for some reason, the root directory of this project is not being added to the python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from model.lib.modules import FeedForward, CausalMultiHeadAttention, RMSNorm
import torch.nn.functional as F
from config.cfg import GPT2Config
from contextlib import nullcontext
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from data.data_common import write_tokenizer


"""
This script implements vanilla transformer model which uses
fixed positional encoding and causal multi-head attention.
"""

# First construct a transformer block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalMultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
            rope_en = cfg["rope_en"],
            theta_ref = cfg["theta_ref"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x  = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
# Now construct the GPT decoder
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.cfg = cfg

        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # initialize the parameters
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init) # .apply() recursively apply _init function to the Model
    
    def _init(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02, generator=self.init_rng)

    def forward(self, in_idx, targets=None, return_logits=True):
        '''
        if targets=-1, then loss is not computed but all logits for all elements in seq are returned
        '''
        batch_size, seq_len = in_idx.shape
        tok_embds = self.tok_emb(in_idx)
        pos_embds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embds + pos_embds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # check if it is in train mode or inference mode
        # check if it is in train mode or inference mode
        if targets is not None:
            logits = self.out_head(x)
            if targets.ndim > 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                loss = None
        elif return_logits:
            # inference time mini-optimization, only fwd the lm_head on the very last position
            logits = self.out_head(x[:,[-1],:]) # Note: using list [-1] to preserve the time dimension
            loss = None
        

        # for some performance reasons
        if not return_logits:
            logits = None
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight matrices, embedding will have weight decay, while biases, layernomr, rmsnorm wont
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1: # use sharding of optimizer states
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0],
                                                optimizer_class = torch.optim.AdamW,
                                                lr=learning_rate,
                                                betas=betas,
                                                fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        
        return optimizer

    @torch.no_grad()
    def generate(self, start_token_idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = start_token_idx
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_seq_length
            idx_cond = idx if idx.size(1) <= self.cfg["context_length"] else idx[:, -self.cfg["context_length"]]
            # make forward pass
            logits, _ = self(idx_cond)
            # pluck the logit at the final step and scale by the desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("Inf")
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            ids_next = torch.multinomial(probs, num_samples=1)
            # append the sampled index to the running sequence and continue
            idx = torch.cat((idx, ids_next), dim=1)
        
        return idx
                                            

    
def print0(*args, **kwargs):
    '''
    This will print only if the node is master for multi-node set up
    '''
    if int(os.environ["RANK"])== 0:
        print(*args, **kwargs)

# learning rate decay scheduler (cosine with warmpup)
def get_lr(it, args):
    min_lr = args.learning_rate * args.learning_rate_decay_frac
    #1: linear warmup for warmup_iters_steps
    if it < args.warmup_iters:
        return args.learning_rate * (it+1) / args.warmup_iters

    #2: if it > args.num_iterations, return min learning rate
    if it > args.num_iterations:
        return min_lr
    
    #3: args.warmup_iters < it <= args.num_iterations: then decay using a cosine schedule
    decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
    assert decay_ratio <= 1.0, "Invalid learning rate decay schedule..."
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (args.learning_rate - min_lr)


# --------- own distributed data loader -----------------------
def _peak_data_shard(filename):
    with open(filename, "rb") as f:
        # only read the header and return header data, header length is 256 elements each having 4 bytes
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    # check if header is valid
    if header[0] != 20250211:
        print("ERROR: magic number mismatch in the data.bin file!: {filename}")
        print("Check whether the file is correct or not")
        exit(1)
    assert header[1] == 1, f"version mismatch in header for file: {filename}"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, header length is 256 elements each having 4 bytes
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20250211, f"ERROR: magic number mismatch in the data.bin file!: {filename}"
        assert header[1] == 1, f"version mismatch in header for file: {filename}"
        ntok = header[2]
        # the rest of data is uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    # check whether ntok matches with len(tokens)
    assert len(tokens) == ntok, f"number of tokens read does not match with token count in header: {filename}"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        '''
        [1] filename_pattern: based on this filename pattern, we will search all the files having the same extension as filename_pattern in
            the same path as filename_pattern
        [2] B: batch size per GPU
        [3] T: max seq lenght
        [4] process_rank: global rank
        [5] num_processes: total number of GPUs
        '''
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        
        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        print0(self.files)
        assert len(self.files) > 0, f"did not find any files that match the pattern: {filename_pattern}"

        # load and count the number of tokens in all shards
        ntok_total = 0.0
        for file in self.files:
            shard_ntok = _peak_data_shard(file)
            assert shard_ntok >= self.num_processes * B * T + 1 # Sachin: ensuring that each .bin file has atleast one batch equivalent tokens for all GPUs (+1 for target generation)
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # if shard 0 is loaded, just reset the current pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1] # +1 for getting targets
        # convert from np to torch tensors
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard 
        self.current_position += B * T * self.num_processes
        # if the current shard does not have required # of tokens to be loaded across all GPU as next batch, then move to next shard (skip the remaining tokens in current shard)
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # move to next shard
            self.advance()
        return x, y


# -------------------------------------------------------------
    
if __name__ == "__main__":
    
    import time
    import argparse
    import tiktoken

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_train_bin", type=str, default="data/tinyshakespare/tinyshakespare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin for eval validation loss")
    parser.add_argument("--output_dir", type=str, default="", help="output directory for writing logs and saving checkpoint")
    parser.add_argument("--model", type=str, default="gpt2", help="currently set it to only gpt2 which is supported, gpt2|d12|d24|d36|d48") # d12: depth12 for gpt2
    parser.add_argument("--load_chkpoint_file", type=str, default="", help="file path where to find the checkpoint file")
    # token layout for each step of optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="max seq length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens") # total_batch_size across all GPUs across all machines ( this will be used in calculating the gradient accumuation steps)
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmp up iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate decay factor")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=100, help="how many steps to compute validation loss") # val loss every kth iteration
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average") # for every val iteration, how many batches of val dataset
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from model") #  perform inference step once every k steps
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensor cores") # ?? Sachin
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we auto-detect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)") # if you choose 1 => shard the optimizer states across GPUs
    # python -> disk
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    parser.add_argument("--save_every_niter", type=int, default=100, help="how many iterations to save the model")
    parser.add_argument("--wandb_log_iters", type=int, default=10, help="how many iteration should wandb log train loss")
    parser.add_argument("--wandb_project_id", type=str, default="", help="project id for weight and baises")
    args = parser.parse_args()

    # args validation
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in  {"gpt2"}

    # set up DDP, torchrun set this variable
    ddp = int(os.environ.get("RANK", -1)) != -1 # whether its a DDP run or not
    if ddp:
        # use of ddp needs CUDA at the moment, we set the rank and local rank
        assert torch.cuda.is_available(), "for now I think DDP needs CUDA"
        init_process_group(backend="nccl") # nccl, glue, mpi : these are high compute comm libraries
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process starts with the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        zero_stage = 0
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            device = args.device
        else:
            # attempt to auto-detect
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.mps.is_available():
                device = "mps"

    print0(f"Using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate the gradient accumulation from the desired total_batch_size
    tokens_per_fwdbwd = B * T * ddp_world_size # number of tokens processed across all GPUS across all nodes (machines)
    #assert args.total_batch_size % tokens_per_fwdbwd == 0 # ?? Sachin, don't need this assert
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # set up seed for re-producability
    torch.manual_seed(42 + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + seed_offset)
    
    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    if master_process and args.write_tensors: # tokenizer is technically not tensor but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # initialize the model with its associated cfg
    gptcfg = asdict(GPT2Config())
    gptcfg["context_length"] = args.sequence_length
    model = GPTModel(gptcfg)
    print0(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    # put model in train mode
    model.train()
    model.to(device)

    if args.compile:
        print0("compiling model ...")
        # below 2 lines, enables inductor (which is backend for torch.compile, the backend converts pyTorch code to optimized Triton kernels)
        # to optimize the code even further
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.coordinate_descent_check_all_directions = True
        model = torch.compile(model)

    # ------ Distributed data sampler ---------------
    train_loader = DistributedDataLoader(args.input_train_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------Distributed sampler code ends ----------------

    # ------ Main training loop ----------------------
    if ddp:
        # wrap the model in DDP
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model # contains the "raw" unwrapped model

    ## init optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, 
                                                learning_rate=args.learning_rate, 
                                                betas = (0.9, 0.95),
                                                device_type = device_type, 
                                                zero_stage = zero_stage)

    ## load the checkpoint if enabled
    if args.load_chkpoint_file:
        chkpoint_file = args.load_chkpoint_file
        print0(f"loading checkpoint from file: {chkpoint_file}")
        state = torch.load(chkpoint_file)
        raw_model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_iter = state['n_iter']
        wandb_run_id = state['wandb_run_id']
        lossf = state['train_loss']
        val_loss = state['val_loss']
        start_iter = global_iter  
        print0(f"start_iter: {start_iter}/{args.num_iterations}")
    else:
        start_iter = 0
        wandb_run_id = None

    # initialize wandb
    if master_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project = args.wandb_project_id, #"pytorch-gpt2-ddp",
            # allow resuming existing run with the same name (in case the rank 0 node crashed)
            name=f"global_rank_{ddp_rank}",
            id=wandb_run_id,
            resume="allow",
            config=vars(args) # args is a Namespace of argparse.Namespace, you can access the dict in namespace by wrapping it in vars(namespace)
        )
    
    if master_process:
        # define our custom x axis metric
        wandb.define_metric("global_step")
        # define which metrics will be plotted against it
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")
    
    # create the loggign DIR if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" and wipe it clean
        with open(logfile, "w") as f:
            pass

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    timings = []
    norm = -1.0 # dummy value to print in inference-only mode
    val_loss = None
    for step in range(start_iter, args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # save the model and optimizer
        if (last_step or (step > 0  and (step % args.save_every_niter) == 0)) and master_process:
            model_filename = os.path.join(args.output_dir,"chkpoint") + f"_{step}.pth"
            # save the model and optimizer
            if args.zero_stage > 0:
                optimizer.consolidate_state_dict() # required in sharded optimizer settings
            torch.save({
                'n_iter': step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id,
                # saving the train and val loss which will be used if we resume from a checkpoint
                'train_loss': lossf,
                'val_loss': val_loss
            }, model_filename)
            print0(f"chkpoint saved at iter: {step}")


        # once in a while evaluate the validation dataset
        if  (args.val_loss_every > 0 and \
            ((step % args.val_loss_every) == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps): # number of val batches
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
                # log to console and to file
                print0(f"step {step+1:4d}/{args.num_iterations} | val loss: {val_loss}")
                if master_process and logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d vloss:%f\n" % (step, val_loss))
                # log to wandb
                if master_process:
                    wandb.log({'val/val_loss': val_loss, 'global_step': step-1})

        # once in a while perform model inference on the master process
        if args.sample_every > 0 and ((step % args.sample_every) == 0 or last_step) and master_process:
            model.eval()
            # we will kickoff the generation with "<|endoftext|>", which designates the start of a new seqeunce
            start_id = [enc.eot_token]
            xg = torch.tensor(start_id, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0("---------------------------------")
            print0(enc.decode(yg[0].tolist()))
            print0("---------------------------------")

        # log train loss on wandb
        if master_process and (step > 0 and (step % args.wandb_log_iters) == 0):
            wandb.log({'train/train_loss': lossf, 'global_step': step-1})
            

        # NOTE: we want to run the eval and inference at 0th iteration and iteration = num_iteration, so we break here
        if last_step:
            break

        # ------- model training section -------------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps (for logging purposes)
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx: # this is torch.amp.autocast for automatic mixe precision , enable autocast for forward pass
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only: # Sachin ?? don't know y this check
                loss.backward() # NOTE: we are not using lossf but loss.backward()
        # we need to all reduce the lossf so that all GPUs have same average value accumulated over the micro-batches across all GPUs
        if ddp:
            lossf = torch.tensor(lossf, requires_grad=False).to(x.device)
            # Remember: all_reduce is getting value from all GPUs and then averaging based on op set below and then broadcasting it back to all GPUs
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item() # for logging purposes
        # clip the gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # run the scheduler to get the learning rate and then update learning rate in param group
        lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # ------- model training section ends --------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    
    # print the average of last 20 timings to get something smoothish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # ------------Main training loop ends---------------------------
    # clean up nice
    if ddp:
        destroy_process_group()



    


