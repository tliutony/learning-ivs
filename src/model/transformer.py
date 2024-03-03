import torch
import torch.nn as nn
import copy


class TransformerEncoder(nn.Module):
    """
    Transformer encoder - consists of repeated TransformerEncoderBlocks.

    n_layers: number of encoder layers
    """
    def __init__(self, n_layers, n_heads, emb_dim):
        super().__init__()
        self.encoder = TransformerEncoderLayer(n_heads, emb_dim)
        self.layers = nn.Sequential([copy.deepcopy(self.encoder) for _ in range(n_layers)])

    def forward(self, x):
        # x is (batch_size, seq_len, emb_dim)
        x = self.layers(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single encoder block in a transformer. Consists of two sublayers, a multi-head attention layer and a 
    fully connected linear layer. Layer Norm, dropout, then summation with the input to the sublayer
    as a residual is done in sequence to the output of each sublayer. TODO: CHECK THIS
    """
    def __init__(self, n_heads, emb_dim) -> None:
        super().__init__()
        self.mh_attn = MultiHeadAttentionBlock(n_heads, emb_dim)
        self.linear = nn.Linear()

    def forward(self, x):
        # x is (batch_size, seq_len, emb_dim)
        x = self.mh_attn(x)



class MultiHeadAttentionBlock(nn.Module):
    """
    A single multi-head attention block. Performs multiple parallelized self-attention weightings,
    concatenates the results across all attention heads, then passes through a fully connected layer (?)
    """
    def __init__(self, emb_dim) -> None:
        super().__init__()



class SelfAttentionBlock(nn.Module):
    """
    Single attention block. Transforms input through three linear transforms to produce query, key, value
    data. Takes pairwise inner products between queries and keys, then normalize via softmax. 

    For simplicity, I've coded it up so that the query/key vector dimensions = value vector dimension
    """
    def __init__(self, emb_dim, qkv_dim) -> None:
        super().__init__()
        # x is (batch_size, seq_len, emb_dim)
        self.W_q = nn.Parameter(torch.zeros_like(emb_dim, qkv_dim))
        self.W_k = nn.Parameter(torch.zeros_like(emb_dim, qkv_dim))
        self.W_v = nn.Parameter(torch.zeros_like(emb_dim, qkv_dim))

    def forward(self, x) -> torch.tensor:
        # x is (batch_size, seq_len, emb_dim)
        d_k = self.W_q.shape(1)
        Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v
        attn_scores = Q @ K.T
