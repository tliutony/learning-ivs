import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer encoder - consists of repeated TransformerEncoderBlocks.
    """
    def __init__(self, n_layers):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.layers = nn.Sequential([])

    def forward(self, x):


class TransformerEncoderBlock(nn.Module):
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
        x = self.mh_attn(x)



class MultiHeadAttentionBlock(nn.Module):
    """
    A single multi-head attention block. Performs multiple parallelized self-attention weightings,
    concatenates the results across all attention heads, then passes through a fully connected layer (?)
    """
    def __init__(self) -> None:
        super().__init__()


class SelfAttentionBlock(nn.Module):
    """
    Single attention block. Transforms input through three linear transforms to produce query, key, value
    data. Applies
    """
