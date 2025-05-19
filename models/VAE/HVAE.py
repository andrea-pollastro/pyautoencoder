import torch
import torch.nn as nn
from typing import List, Tuple
from utils.loss import hierarchical_ELBO

class HierarchicalVAE():
    def __init__(self,
                 encoders: List[nn.Module],
                 decoders: List[nn.Module],
                 latent_dims: List[int]):
        super().__init__()

        assert len(encoders) == len(decoders) == len(latent_dims), \
            "Mismatch in number of encoders, decoders, or latent_dims"
