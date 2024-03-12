"""### identity.py
###### in `mlx_nerf/encoding`

No encoding.
"""


import mlx.core as mx
import mlx.nn as nn

from mlx_nerf.encoding import Encoding

class IdentityEncoding(Encoding):
    def __init__(
        self, 
        in_dim: int, 
    ) -> None:
        
        super().__init__(in_dim)
        return


    def get_out_dim(self):

        return self.in_dim
    
    def forward(
            self, 
            in_array: mx.array
        ):

        return in_array