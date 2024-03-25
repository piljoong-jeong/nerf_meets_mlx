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

        # NOTE: all the coefficients below are constants, 
        # NOTE: appeared during evaluation of https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        # NOTE: here, `r=1` as `in_dirs` is unit vector set
        level = self.n_degrees
        if level >= 0:
            # NOTE: 1/2 * sqrt(1 / PI)
            out_encoded[...,  0] = 0.28209479177387814 
        if level >= 1:
            # NOTE: sqrt(3 / 4*PI)
            out_encoded[...,  1] = 0.4886025119029199 * y 
            out_encoded[...,  2] = 0.4886025119029199 * z
            out_encoded[...,  3] = 0.4886025119029199 * x
        if level >= 2:
            out_encoded[...,  4] = 1.0925484305920792 * xy
            out_encoded[...,  5] = 1.0925484305920792 * yz
            out_encoded[...,  6] = 0.9461746957575601 * zz - 0.31539156525251999
            out_encoded[...,  7] = 1.0925484305920792 * xz
            out_encoded[...,  8] = 0.5462742152960396 * (xx - yy)
        if level >= 3:
            out_encoded[...,  9] = 0.5900435899266435 * y * (3 * xx - yy)
            out_encoded[..., 10] = 2.890611442640554 * xy * z
            out_encoded[..., 11] = 0.4570457994644658 * y * (5 * zz - 1) # NOTE: 4zz-xx-yy = 5zz-(xx+yy+zz) = 5zz - 1
            out_encoded[..., 12] = 0.3731763325901154 * z * (5 * zz - 3) # NOTE: 2zz-3xx-3yy = 5zz-3(zz+xx+yy) = 5zz - 3
            out_encoded[..., 13] = 0.4570457994644658 * x * (5 * zz - 1) 
            out_encoded[..., 14] = 1.445305721320277 * z * (xx - yy)
            out_encoded[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)
        if level >= 4:
            out_encoded[..., 16] = 2.5033429417967046 * xy * (xx - yy)
            out_encoded[..., 17] = 1.7701307697799304 * yz * (3 * xx - yy)
            out_encoded[..., 18] = 0.9461746957575601 * xy * (7 * zz - 1)
            out_encoded[..., 19] = 0.6690465435572892 * yz * (7 * zz - 3)
            out_encoded[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
            out_encoded[..., 21] = 0.6690465435572892 * xz * (7 * zz - 3)
            out_encoded[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
            out_encoded[..., 23] = 1.7701307697799304 * xz * (xx - 3 * yy)
            out_encoded[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return out_encoded