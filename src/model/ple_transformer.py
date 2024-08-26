import torch
from model import TransformerEncoder
import pytorch_lightning as pl
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

class PLE_Transformer(TransformerEncoder):
    """
    Wrapper around TransformerEncoder, with Piecewise Linear Encoding applied to the output of the encoder
    """
    def __init__(self, n_blocks, n_heads, d_model, d_hidden, dropout: float = 0.1, lr: float = 0.001,
                 weight_decay: float = 0.0) -> None:
        """
        """
        super().__init__()
        # initialize PLE things
        self.ple = PiecewiseLinearEmbeddings() 
        # TODO: paper uses compute_bins on initial dataset to calculate bins, but we're applying them to the output of transformer. we don't exactly have outputs just yet..

    def forward(self, x, padding_mask=None) -> torch.Tensor:
        # x is (batch_size, seq_len, emb_dim)
        context_dict = self.model({'data':x, 'padding_mask':padding_mask}) # dict required since nn.Sequential only takes single arguments as input
        context = context_dict['data']
        # squeeze embedding dimension in context vector, then apply PiecewiseLinearEmbedding
        squeezed_context = torch.squeeze(context) # (bs, seq_len)
        embedded_context = self.ple(squeezed_context) # (bs, seq_len, ple_embed_dim) # does ple do further transforms?
        # still needed? since ple also does linear (i think we should keep this final linear layer)
        # may want to change final_linear to act on entire context rather than last position
        output = self.final_linear(context[:, -1, :])
        return output # shape (bs, 1, 1)