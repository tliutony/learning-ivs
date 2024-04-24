# %%
import torch
import torch.nn as nn
import numpy as np
import copy

# %%
class TransformerEncoder(nn.Module):
    """
    Transformer encoder - consists of `n_blocks` EncoderBlocks in series.
    """
    def __init__(self, n_blocks, n_heads, d_model, d_hidden, dropout=0.1) -> None:
        """
        Initialize transformer encoder.

        n_blocks: number of encoder blocks
        n_heads: number of attention heads in parallel for each MultiHeadAttention sublayer
        d_model: dimension of data throughout attention mechanism (see EncoderBlock for more details)
        d_hidden: dimension of hidden layer in MLP sublayer for each EncoderBlock
        """
        super().__init__()
        self.encoder = EncoderBlock(n_heads, d_model, d_hidden, dropout)
        self.model = nn.Sequential(*[copy.deepcopy(self.encoder) for _ in range(n_blocks)])
        self._initialize_weights(self.model)

    def _initialize_weights(self, model : nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x, padding_mask=None) -> torch.Tensor:
        # x is (batch_size, seq_len, emb_dim)
        x = self.model(x, padding_mask)
        return x

    

# %%
class EncoderBlock(nn.Module):
    """
    Single encoder block in a transformer. Consists of two sublayers, a multi-head attention layer and a MLP with one hidden layer. Residual connections, Layer Norm, and Dropout are used for each sublayer, in sequence shown below:

    input --> LayerNorm --> SubLayer --> Dropout --> (+) --> out
      |_______________________________________________|^
    
    """
    def __init__(self, n_heads:int, d_model:int, d_hidden:int, dropout=0.1) -> None:
        """
        n_heads: number of attention heads working in parallel
        d_model: dimension of inputs and outputs, as well as of intermediate query, key, and value vectors in attention; the input and output dimensions are set to be the same primarily to ensure residual connections work properly
        d_hidden: dimension of hidden layer in MLP
        dropout: dropout probability in dropout layer
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        # assuming v_dim = qk_dim, enforces number of parameters remains unchanged with number of heads (for comparisons)
        assert d_model % n_heads == 0
        qk_dim, v_dim = d_model // n_heads, d_model // n_heads
        self.mh_attn = MultiHeadAttentionBlock(n_heads, in_dim=d_model, qk_dim=qk_dim, v_dim=v_dim, out_dim=d_model, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden),
                                 nn.ReLU(),
                                 nn.Linear(d_hidden, d_model))
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        input: x (batch_size, max_seq_len, d_model)
        output: a (batch_size, max_seq_len, d_model)
        padding_mask: mask to ignore padding tokens when applying softmax to attention matrix. (see MultiHeadAttention forward for more details)
        """
        # MultiHeadAttention sublayer
        z = self.attn_norm(x)
        z, _ = self.mh_attn(x, x, x, padding_mask) # out is (bs, seq, d_model)
        z = x + self.attn_dropout(z) 

        # MLP sublayer
        a = self.mlp_norm(z)
        a = self.mlp(a)
        a = z + self.mlp_dropout(a)
        return a



# %%
class MultiHeadAttentionBlock(nn.Module):
    """
    A single multi-head attention block. Performs multiple parallelized self-attention weightings, concatenates the results across all attention heads, then passes through a fully connected layer (?)
    """
    def __init__(self, n_heads, in_dim, qk_dim, v_dim, out_dim, dropout=0.1) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.W_q = nn.Linear(in_dim, qk_dim * n_heads)
        self.W_k = nn.Linear(in_dim, qk_dim * n_heads)
        self.W_v = nn.Linear(in_dim, v_dim * n_heads)
        self.proj = nn.Linear(v_dim * n_heads, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, padding_mask=None) -> torch.tensor:
        """
        In self attention, queries = keys = values = x

        padding_mask: mask for ignoring certain columns of the raw attention matrix when applying softmax. 'True' corresps. to padding tokens. 
        
        padding_mask is used to set columns corresp. to padded tokens to -1e-9, so they effectively contribute nothing to the softmax and are set to 0. this has the effect of ensuring that padded tokens are not attended to. (rows corresp. to padding are not masked, as they only need to be masked/ignored at loss calculation to ensure they have no effect on the loss and thus learning.)
        """
        # x is (batch_size, seq_len, emb_dim), where seq_len is max sequence length of all sequences in batch
        bs, seq, _ = queries.shape # batch_size, seq_len 
        n_heads = self.n_heads
        qk_dim = self.qk_dim
        v_dim = self.v_dim

        # apply linears to get Q K V vectors
        Q = self.W_q(queries).view(bs, seq, n_heads, qk_dim).transpose(1, 2) # (bs, n_heads, seq, qk_dim)
        K = self.W_k(keys).view(bs, seq, n_heads, qk_dim).transpose(1, 2) # (bs, n_heads, seq, qk_dim)
        V = self.W_v(values).view(bs, seq, n_heads, v_dim).transpose(1,2) # (bs, n_heads, seq, v_dim)

        # get raw attention scores
        attn = Q @ K.transpose(2,3) # (bs, n_heads, seq, seq)

        if padding_mask is not None:
            # (bs, seq, 1) -> (bs, 1, seq, 1) for n_head dimension broadcasting
            padding_mask = padding_mask.unsqueeze(dim=1)
            attn.masked_fill(~padding_mask, -1e9) 
            # see above for explanation of padding masking

        # normalize
        d_k = K.shape[-1]
        attn /= np.sqrt(d_k)
        attn = torch.softmax(attn, dim=-1) # same shape

        # apply dropout to attention scores
        attn = self.dropout(attn)

        # get attended vectors as weighted sum of value vectors
        Z = attn @ V # (bs, n_heads, seq, v_dim)

        # concatenate vectors along heads dimension
        Z = Z.transpose(1,2).reshape(bs, seq, -1) # (bs, seq, n_heads * v_dim)
        # send through linear layer
        out = self.proj(Z) # (bs, seq, out_dim)
        return out, attn 

# %%
