"""### spherical_harmonics.py
###### in `mlx_nerf/encoding`

Encoder of unit vectors into Spherical Harmonics basis
"""


import mlx.core as mx
import mlx.nn as nn

from mlx_nerf.encoding import Encoding

class SphericalHarmonicsEncoding(Encoding):
    def __init__(
        self, 
        in_dim: int, 
        n_degrees: int, 
    ) -> None:
        super().__init__(in_dim)

        self.n_degrees = n_degrees
        
        return

    def get_out_dim(self):
        
        out_dim = self.n_degrees ** 2

        return out_dim
    
    def __call__(
            self, 
            in_array: mx.array # [B, in_dim]
    ):
        """### SphericalHarmonicsEncoding.forward
        ###### in `mlx_nerf/encoding/spherical_harmonics.py`

        """

        # TODO: should I implement spherical harmonics in an independent module?
        # TODO: bc there can be a number of different usage of SH calculations
        
        raise NotImplementedError