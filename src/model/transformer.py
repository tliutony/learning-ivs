# %%
import torch
import torch.nn as nn
import numpy as np
import copy

# %%
class TransformerEncoder(nn.Module):
    """
    Transformer encoder - consists of `n_blocks` repeated EncoderBlocks.

    n_layers: number of encoder layers
    """
    def __init__(self, n_blocks, n_heads, emb_dim):
        super().__init__()
        self.encoder = TransformerEncoderLayer(n_heads, emb_dim)
        self.layers = nn.Sequential([copy.deepcopy(self.encoder) for _ in range(n_layers)])

    def forward(self, x):
        # x is (batch_size, seq_len, emb_dim)
        x = self.layers(x)

# %%
class EncoderBlock(nn.Module):
    """
    Single encoder block in a transformer. Consists of two sublayers, a multi-head attention layer and a fully connected linear layer (or MLP with nonlinearities?). Layer Norm, dropout, then summation with the input to the sublayer as a residual is done in sequence to the output of each sublayer. 
    TODO: CHECK THIS
    """
    def __init__(self, n_heads:int, d_model, d_hidden, dropout=0.1) -> None:
        """
        n_heads: 
        """
        super().__init__()
        self.mh_attn = MultiHeadAttentionBlock(n_heads, in_dim=d_model, qk_dim=d_model, v_dim=d_model, out_dim=d_model, attn_dropout=dropout)
        self.mlp = nn.Sequential([nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear()]) # single linear layer? or MLP?
        self.attn_dropout, self.mlp_dropout = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        # x is (batch_size, seq_len, in_dim)
        # maybe should include functionality for different types of attention - for now we'll just assume self-attention
        z, attn = self.mh_attn(x, x, x) # out is (bs, seq, out_dim)
        z = self.dropout1(x)
        z = z + x # needs input and output of attention to have the same size, i.e in_dim = out_dim???
        a = self.fc(z)
        a = self.dropout2(a)
        a = a + z
        # Layer norm?



# %%
class MultiHeadAttentionBlock(nn.Module):
    """
    A single multi-head attention block. Performs multiple parallelized self-attention weightings, concatenates the results across all attention heads, then passes through a fully connected layer (?)
    """
    def __init__(self, n_heads, in_dim, qk_dim, v_dim, out_dim, attn_dropout=0.1) -> None:
        super().__init__()
        # assuming v_dim = qk_dim, enforces number of parameters remains unchanged with number of heads (for ablation/comparisons)
        assert in_dim == n_heads * qk_dim
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.W_q = nn.Linear(in_dim, qk_dim * n_heads)
        self.W_k = nn.Linear(in_dim, qk_dim * n_heads)
        self.W_v = nn.Linear(in_dim, v_dim * n_heads)
        self.fc = nn.Linear(v_dim * n_heads, out_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, queries, keys, values) -> torch.tensor:
        """
        In self attention, queries = keys = values = x
        """
        #TODO: masking???
        # x is (batch_size, seq_len, emb_dim)
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

        # normalize
        d_k = K.shape[-1]
        attn /= np.sqrt(d_k)
        attn = torch.softmax(attn, dim=-1) # same shape

        # apply dropout to attention scores
        attn = self.attn_dropout(attn)

        # get attended vectors as weighted sum of value vectors
        Z = attn @ V # (bs, n_heads, seq, v_dim)

        # concatenate vectors along heads dimension
        Z = Z.transpose(1,2).reshape(bs, seq, -1) # (bs, seq, n_heads * v_dim)
        # send through fc linear layer
        out = self.fc(Z) # (bs, seq, out_dim)
        return out, attn 
        # maybe look at why we return attention too?





        



# class SelfAttentionBlock(nn.Module):
#     """
#     Single attention block. Transforms input through three linear transforms to produce query, key, value
#     data. Takes pairwise inner products between queries and keys, then normalize via softmax. 

#     For simplicity, I've coded it up so that the query/key vector dimensions = value vector dimension
#     """
#     def __init__(self, emb_dim, qkv_dim) -> None:
#         super().__init__()
#         # x is (batch_size, seq_len, emb_dim)
#         self.W_q = nn.Parameter(torch.zeros_like(emb_dim, qkv_dim))
#         self.W_k = nn.Parameter(torch.zeros_like(emb_dim, qkv_dim))
#         self.W_v = nn.Parameter(torch.zeros_like(emb_dim, qkv_dim))

#     def forward(self, x) -> torch.tensor:
#         """
#         x is (batch_size, seq_len, emb_dim)
#         returns: (batch_size, seq_len, qkv_dim) scaled dot-product attention-reweighted inputs
#         """
#         d_k = self.W_q.shape[1]
#         Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v  # (bs, seq, qkv_dim)
#         # give these better names
#         raw_attn = torch.bmm(Q, K.mT) # (bs, seq, seq)
#         norm_attn = torch.softmax(raw_attn/np.sqrt(d_k), dim=1) # normalize, shape unchanged
#         head = torch.bmm(norm_attn, V) # (bs, seq, qkv_dim)
#         return head

# %%
