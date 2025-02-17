import torch
import torch.nn as nn
import torchtune

from typing import Optional
from einsum import einsum

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/ torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.bias
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
## Feed-Forward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
# Causal MulitHead Attention Implementation
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, 
                 qkv_bias=False, rope_en=False, theta_ref=50000):
        super().__init__()
        assert d_in % num_heads == 0, "emb_dim must be divisible by num_heads"
    
        self.num_heads = num_heads
        self.max_seq_len = context_length
        self.rope_en = rope_en
        self.theta_ref = theta_ref
        self.dropout_val = dropout

        self.head_dim = d_in // self.num_heads
        if self.rope_en:
            """
            We use torchtune implementation instead of our implementation
            Just assuming its more efficient
            """
            self.rope_emb = torchtune.modules.RotaryPositionalEmbeddings(
                self.head_dim, self.max_seq_len, self.theta_ref
            )
        else:
            self.rope_emb = None

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.Wo = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # flash attention support is there only for Pytorch >=2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention, Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("mask",
                    torch.triu(torch.ones(context_length, context_length), diagonal=1).view(
                        1, 1, context_length, context_length
                    ))
    
    def forward(self, x):
        bs, num_tokens, d_in = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # (b, n_tokens, emb_dim) -> (b, n_tokens, n_head, head_dim)
        q = q.view(bs, num_tokens, self.num_heads, -1)
        k = k.view(bs, num_tokens, self.num_heads, -1)
        v = v.view(bs, num_tokens, self.num_heads, -1)

        if self.rope_en:
            # apply ROPE to query and key
            q = self.rope_emb(q) # (b, n_tokens, n_heads, head_dim)
            k = self.rope_emb(k) # (b, n_tokens, n_heads, head_dim)

        # split into heads (b, n_tokens, n_heads, head_dim) -> (b, n_heads, n_tokens, head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if self.flash:
            # (bs, n_heads, n_tokens, head_dim)
            context_vec = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                           attn_mask=None, 
                                                                           dropout_p=self.dropout_val, 
                                                                           is_causal=True)
        else:

            # compute attn_scores
            attn_scores = q @ k.transpose(2,3)

            # original mask truncated to num_tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # use mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights) # (bs, n_heads, n_tokens, n_tokens)

            context_vec = (attn_weights @ v) # (bs, n_heads, n_tokens, head_dim)

        context_vec = context_vec.transpose(1,2) # (bs, n_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(bs, num_tokens, -1) # (bs, n_tokens, d_out)
        
        return context_vec, attn_weights
    

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int):
        """
        dim: dim of each head
        max_seq_len: maximum seq len
        base: this is theta_ref (like 50000)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # initialize the cache 
        self._rope_init()

    def _rope_init(self):
        theta = self.base ** -(torch.arange(0, self.dim, 2)/self.dim)
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
    
class RMSNorm(torch.nn.Module):
    """
    Implements the RMS Normalization (Root Mean Square Normalization) layer.
    RMSNorm is a variant of layer normalization that normalizes the activations
    of the previous layer based on their root mean square value.

    Parameters:
    - dim (int): The dimension of the input features the normalization is applied to.
    - eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
    - add_unit_offset (bool): If True, adds a unit (1) to the learned scaling coefficient, effectively
      starting with no scaling. If False, the scaling coefficient starts from zero. Default is True.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__() 
        self.eps = eps  # Small epsilon value for numerical stability since you can't divide by 0
        self.add_unit_offset = add_unit_offset  # Flag to determine if a unit should be added to the weight
        
        # Initialize the weight parameter with zeros, which will be learned during training.
        # The shape of the weight is [dim], meaning one weight per feature dimension.
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Private helper function to normalize the input tensor.

        Parameters:
        - x (Tensor): The input tensor to normalize.

        Returns:
        - Tensor: The normalized tensor.
        """
        # Calculate the root mean square value for each feature (across the last dimension),
        # then use reciprocal square root (rsqrt) for normalization.
        # Add self.eps to the denominator for numerical stability.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass of the RMSNorm layer.

        Parameters:
        - x (Tensor): The input tensor to normalize.

        Returns:
        - output: The normalized and scaled tensor.
        """
        # Normalize the input tensor using the _norm function and ensure the data type matches the input.
        x = self._norm(x.float()).type_as(x)
        
        # If add_unit_offset is True, scale the normalized tensor by (1 + self.weight),
        # effectively starting with a scaling factor of 1 (no scaling).
        # Otherwise, scale by self.weight only.
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
            
        # Return the scaled output tensor.
        return output

