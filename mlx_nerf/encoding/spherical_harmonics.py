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
        n_degrees: int, # NOTE: identical to SH level; [0, 4]
    ) -> None:
        super().__init__(in_dim)

        assert 0<=n_degrees<=4, f"[ERROR] {n_degrees=} must be in range [0, 4]!"

        self.n_degrees = n_degrees
        
        return

    def get_out_dim(self):
        
        out_dim = (self.n_degrees+1) ** 2

        return out_dim
    
    def __call__(
        self, 
        in_dirs: mx.array # [B, in_dim]
    ):
        """### SphericalHarmonicsEncoding.forward
        ###### in `mlx_nerf/encoding/spherical_harmonics.py`

        """

        # TODO: should I implement spherical harmonics in an independent module?
        # TODO: bc there can be a number of different usage of SH calculations

        x = in_dirs[..., 0]
        y = in_dirs[..., 1]
        z = in_dirs[..., 2]

        xx = x*x
        yy = y*y
        zz = z*z
        xy = x*y
        yz = y*z
        xz = x*z

        out_encoded = mx.zeros((*in_dirs.shape[:-1], self.get_out_dim()), dtype=in_dirs.dtype)

        # NOTE: by evaluating https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        # NOTE: here, `r=1` as `in_dirs` is unit vector set
        level = self.n_degrees
        if level >= 0: # NOTE: for readability
            out_encoded[..., 0] = 0.28209479177387814
        if level >= 1:
            out_encoded[..., 1] = 0.4886025119029199 * y
            out_encoded[..., 2] = 0.4886025119029199 * z
            out_encoded[..., 3] = 0.4886025119029199 * x
        if level >= 2:
            out_encoded[..., 4] = 1.0925484305920792 * xy
            out_encoded[..., 5] = 1.0925484305920792 * yz
            out_encoded[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
            out_encoded[..., 7] = 1.0925484305920792 * xz
            out_encoded[..., 8] = 0.5462742152960396 * (xx - yy)
        if level >= 3:
            # TODO
            pass
        if level >= 4:
            # TODO
            pass

        raise NotImplementedError