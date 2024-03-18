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
        min_freq_exp: float = None, 
        max_freq_exp: float = None, 
        is_include_input: bool = False, 
    ) -> None:
        super().__init__(in_dim)

        self.n_freqs = n_freqs
        self.min_freq_exp = min_freq_exp if min_freq_exp else 0.0
        self.max_freq_exp = max_freq_exp if max_freq_exp else float(n_freqs-1)

        self.is_include_input = is_include_input
        return

    def get_out_dim(self):
        out_dim = self.in_dim * self.n_freqs * len([mx.sin, mx.cos]) # NOTE: `in_dim=3` -> 3 * (2*10) + 3 => 63
        
        if self.is_include_input:
            out_dim += self.in_dim

        return out_dim
    
    def __call__(
            self, 
            in_array: mx.array
    ):
        """### SinusoidalEncoding.forward
        ###### in `mlx_nerf/encoding/sinusoidal.py`

        Implementation of Eq. (4) in NeRF [ECCV2020]
        """

        freq_bands = 2.0 ** mx.linspace(
            self.min_freq_exp, self.max_freq_exp, num=self.n_freqs
        )
        in_array_scaled = in_array
        # in_array_scaled = 2.0 * mx.pi * in_array # NOTE: this performs worse in image training; TODO: see if the degeneration occurs in volume learning as well
        in_array_scaled = in_array_scaled[..., None] * freq_bands # [B, in_dim, n_freqs]
        in_array_scaled = mx.reshape(in_array_scaled, (in_array_scaled.shape[0], -1)) # [B, in_dim * n_freqs]

        out_encoded = mx.sin(mx.concatenate(
            [
                in_array_scaled,                # NOTE: sin
                in_array_scaled + mx.pi / 2.0,  # NOTE: cos; due to minimize memory access
            ], axis=-1
        )) # [B, 2 * in_dim * n_freqs]

        if self.is_include_input:
            out_encoded = mx.concatenate([out_encoded, in_array], axis=-1)

        return out_encoded