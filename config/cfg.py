from dataclasses import dataclass, field

@dataclass(frozen=True)
class GPT2Config:
    '''
    This will be vanilla GPT Config with fixed Positional Encoding, Causal MHA
    '''
    vocab_size:int = field(default=50257)
    context_length:int = field(default=1024)
    emb_dim:int = field(default=768)
    n_heads:int = field(default=12)
    n_layers:int = field(default=12)
    drop_rate:float = field(default=0.1)
    qkv_bias:bool = field(default=False)
    rope_en: bool = field(default=False)
    theta_ref:int = field(default=50000) # only valid when rope_en is TRUE


if __name__ == "__main__":
    gpt2cfg = GPT2Config()
    print(gpt2cfg)
