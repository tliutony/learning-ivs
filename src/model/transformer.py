# %%
import torch
import torch.nn as nn
import numpy as np
import copy
from einops import rearrange, reduce
import pytorch_lightning as pl
from typing import Optional

PADDING_VALUE = -1e9
# %%
class TransformerEncoder(pl.LightningModule):
    """
    Transformer encoder - consists of `n_blocks` EncoderBlocks in series.
    """
    def __init__(self, n_blocks, n_heads, d_model, d_hidden, dropout: float = 0.1, lr: float = 0.001,
                 weight_decay: float = 0.0, pooling: Optional[str] = None, seq_len: Optional[int] = None, qk_dim: Optional[int] = None, v_dim: Optional[int] = None) -> None:
        """
        Initialize transformer encoder.

        n_blocks: number of encoder blocks
        n_heads: number of attention heads in parallel for each MultiHeadAttention sublayer
        d_model: dimension of data throughout attention mechanism (see EncoderBlock for more details)
        d_hidden: dimension of hidden layer in MLP sublayer for each EncoderBlock
        pooling: specifies how output of final layer across all tokens are aggregated. if None, then concatenates all outputs and applies a Linear. otherwise, one of {'average', ???}
        seq_len: length of input sequence. only required if pooling is None, since in order to apply a Linear to concatenation of all vectors in a sequence, one needs to know the length of the sequence. assumes all input has fixed length.
        qk/v_dim: see EncoderBlock docs
        """
        super().__init__()
        self.encoder = EncoderBlock(n_heads, d_model, d_hidden, dropout, qk_dim, v_dim)
        self.model = nn.Sequential(*[copy.deepcopy(self.encoder) for _ in range(n_blocks)])
        self._initialize_weights(self.model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.pooling = pooling
        # now this depends on how we pool, which depends on the size of the model
        if self.pooling is None: # concatenate final layer outputs for all tokens, apply linear to get scalar prediction
            assert seq_len is not None, "seq_len argument must be defined when pooling is None (see docstrings)"
            self.final_linear = nn.Linear(seq_len * d_model, 1)
        else: # pool according to some pooling/aggregation scheme
            self.final_linear = nn.Linear(d_model, 1)

    def _initialize_weights(self, model : nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x, padding_mask=None) -> torch.Tensor:
        # x is (batch_size, seq_len, emb_dim)
        context_dict = self.model({'data':x, 'padding_mask':padding_mask}) # dict required since nn.Sequential only takes single arguments as input
        context = context_dict['data']
        # aggregate over all layers
        if self.pooling == 'average': # average pooling
            averaged_context = reduce(context, "bs seqlen d_model -> bs d_model", 'mean')
            output = self.final_linear(averaged_context)
        elif self.pooling == 'attention':
            pass  # TODO: implement this later
        else: # self.pooling is None, concatenate along sequence length dimension
            concatenated_context = rearrange(context, "bs seqlen d_model -> bs (seqlen d_model)")
            output = self.final_linear(concatenated_context)
        # padding_mask = context_dict['padding_mask'] # in case you need it 
        return output # shape (bs, 1, 1)
    
    # copied from other methods - modify as needed
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'val')

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'test')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.0)
        return [optimizer], [scheduler]

    def step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        """
        Forward pass of the network.
        Args:
            batch: The input data.
            stage: The stage of the network.
        Returns:
            The output of the network.
        """
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.MSELoss()(y_hat, y)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss
    

# %%
class EncoderBlock(nn.Module):
    """
    Single encoder block in a transformer. Consists of two sublayers, a multi-head attention layer and a MLP with one hidden layer. Residual connections, Layer Norm, and Dropout are used for each sublayer, in sequence shown below:

    input --> LayerNorm --> SubLayer --> Dropout --> (+) --> out
      |_______________________________________________|^
    
    """
    def __init__(self, n_heads:int, d_model:int, d_hidden:int, dropout=0.1, qk_dim: Optional[int] = None, v_dim: Optional[int] = None) -> None:
        """
        n_heads: number of attention heads working in parallel
        d_model: dimension of inputs and outputs, as well as of intermediate query, key, and value vectors in attention; the input and output dimensions are set to be the same primarily to ensure residual connections work properly
        d_hidden: dimension of hidden layer in MLP
        dropout: dropout probability in dropout layer
        qk/v_dim: specify dimension of query, key / value vectors
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)

        def default_qkv_dim(d_model, n_heads):
            """
            returns default dimension of query, key / value vectors, given d_model and number attention heads
            """
            if d_model < n_heads:
                return d_model
            else:
                return d_model // n_heads
            
        if qk_dim is None:
            qk_dim = default_qkv_dim(d_model, n_heads)
        if v_dim is None:
            v_dim = default_qkv_dim(d_model, n_heads)
        self.mh_attn = MultiHeadAttentionBlock(n_heads, in_dim=d_model, qk_dim=qk_dim, v_dim=v_dim, out_dim=d_model, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden),
                                 nn.ReLU(),
                                 nn.Linear(d_hidden, d_model))
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, input_dict):
        """
        input: inpuct_dict, {'data': x, 'padding_mask':padding_mask}, where
            
            x: a (batch_size, max_seq_len, d_model) data tensor
            padding_mask: mask to ignore padding tokens when applying softmax to attention matrix. (see MultiHeadAttention forward for more details)

            inputs are packaged this way to allow EncoderBlock to be used with nn.Sequential, which expects atoms to have only a single input

        output: a (batch_size, max_seq_len, d_model)
        """
        # extract items from dictionary
        x = input_dict['data']
        padding_mask = input_dict['padding_mask']

        # MultiHeadAttention sublayer
        z = self.attn_norm(x)
        z, _ = self.mh_attn(x, x, x, padding_mask) # out is (bs, seq, d_model)
        z = x + self.attn_dropout(z) 

        # MLP sublayer
        a = self.mlp_norm(z)
        a = self.mlp(a)
        a = z + self.mlp_dropout(a)
        return {'data':a, 'padding_mask':padding_mask}

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
        
        padding_mask is used to set columns corresp. to padded tokens to -1e9, so they effectively contribute nothing to the softmax and are set to 0. this has the effect of ensuring that padded tokens are not attended to. (rows corresp. to padding are not masked, as they only need to be masked/ignored at loss calculation to ensure they have no effect on the loss and thus learning.)
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
            attn.masked_fill(~padding_mask, PADDING_VALUE) 
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
