import torch
import numpy as np

from typing import Optional

class SinusoidalPositionalEmbedding(torch.nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: torch.nn.Parameter) -> torch.nn.Parameter:
        """
        Interleaved sine and cosine position embeddings
        """
        out.requires_grad = False
        out.detach_()
        
        N, D = out.shape

        ## TODO: Create a N x D//2 array of position encodings (argument to the sine/cosine)
        inds = np.arange(0, D // 2)
        k = np.arange(N)
        denom = 1 / np.power(10_000, 2*inds / D)
        position_enc = np.outer(k, denom)  # Efficiently make N x D//2 array for all positions/dimensions
        #####

        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc))  # Even indices get sin
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc))  # Odd indices get cos
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)