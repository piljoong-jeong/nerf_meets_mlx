"""### sinusoidal.py
###### in `mlx_nerf/encoding`

Multi-level sinusoidal encoding. First presented in dear Transformers [NIPS2017], and adapted in dear NeRF [ECCV2020]
"""


import mlx.core as mx
import mlx.nn as nn

from mlx_nerf.encoding import Encoding

class SinusoidalEncoding(Encoding):
    def __init__(
        self, 
        in_dim: int, 
        n_freqs: int, 
        min_freq: float, 
        max_freq: float, 
        is_include_input: bool = False, 
    ) -> None:
        super().__init__(in_dim)

        self.n_freqs = n_freqs
        self.min_freq = min_freq if min_freq else 0
        self.max_freq = max_freq

        self.is_include_input = is_include_input
        

    def get_out_dim(self):
        out_dim = self.in_dim * self.n_freqs * len([mx.sin, mx.cos]) # NOTE: `in_dim=3` -> 3 * (2*10) + 3 => 63
        
        if self.is_include_input:
            out_dim += self.in_dim

        return out_dim
    
    def forward(
            self, 
            in_array: mx.array
        ):
        """### forward
        ###### in `mlx_nerf/encoding/sinusoidal.SinusoidalEncoding`

        Implementation of Eq. (4) in NeRF [ECCV2020]
        """

        in_array_2phi = 2.0 * mx.pi * in_array
        freq_bands = 2.0 ** mx.linspace(
            self.min_freq, self.max_freq, num=self.n_freqs
        )
        in_array_scaled = in_array_2phi[..., None] * freq_bands # [B, in_dim, n_freqs]
        in_array_scaled = mx.reshape(in_array_scaled, (in_array_scaled.shape[0], -1)) # [B, in_dim * n_freqs]

        out_encoded = mx.sin(mx.concatenate(
            [
                in_array_scaled,                # NOTE: sin
                in_array_scaled + mx.pi / 2.0,  # NOTE: cos; due to minimize memory access
            ], axis=-1
        )) # [B, in_dim * n_freqs]

        return out_encoded