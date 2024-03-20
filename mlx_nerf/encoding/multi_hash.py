"""### multi_hash.py
###### in `mlx_nerf/encoding`

Multi-level hash-grid encoding. Presented in Instant Neural Graphics Primitives [SIGGRAPH2022]. Disencouraged to use this for commercial projects, as this algorithm is under proprietary license.
"""


import mlx.core as mx
import mlx.nn as nn

from mlx_nerf.encoding import Encoding

class MultiHashEncoding(Encoding):
    def __init__(
        self, 
        in_dim: int, 
        n_levels: int, 
        min_res: int, # NOTE: coarse resolution
        max_res: int, # NOTE: finest resolution
        n_features_per_level: int, 
        log2_hashmap_size: int, 
        hash_init_scale: float = 0.0001, 
    ) -> None:
        super().__init__(in_dim)

        self.n_levels = n_levels    # NOTE: `L` in Sec. 3
        self.min_res = min_res      # NOTE: `N_min` in Sec. 3
        self.max_res = max_res      # NOTE: `N_max` in Sec. 3
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size

        levels = mx.arange(self.n_levels)

        # NOTE: `b` in Eq. (3)
        self.growing_factor = mx.exp(
            (mx.log(self.max_res) - mx.log(self.min_res)) / (self.n_levels - 1)
        ) if self.n_levels > 1 else 1

        # NOTE: `N_l` in Eq. (2)
        self.scaled_res = mx.floor(self.min_res * (self.growing_factor ** levels)) # [L]

        # NOTE: `T` in Table. (1)
        self.hash_table_size = 2 ** self.log2_hashmap_size # [T]

        # NOTE: construct hash table for each level
        self.hash_table = [
            nn.Embedding(self.hash_table_size, self.n_features_per_level)
            for _ in levels
        ]
        # NOTE: from `Initialization` in Sec. 4
        [nn.init.uniform(-hash_init_scale, hash_init_scale)(table.weight) for table in self.hash_table]    

        return

    def get_out_dim(self):
        out_dim = self.n_levels * self.n_features_per_level

        return out_dim
    
    # TODO: validate
    def hash(
        self, 
        in_array: mx.array # [B, n_levels, in_dim]
    ):
        # NOTE: from [Lehmer 1951]
        list_primes = [
            PRIME1 := 1,          # NOTE: for better cache coherence
            PRIME2 := 2654435761,
            PRIME3 := 805459861,
        ]
        
        out_hashed = mx.zeros_like(in_array)[..., 0] # NOTE: single last dimension, as we'll iteratively apply XOR for each axis
        for i in range(in_array.shape[-1]):
            out_hashed ^= in_array[..., i] * list_primes[i]
        out_hashed %= self.hash_table_size # NOTE: we may try optimize here using &, but not sure if that'll be crucial
    
        return out_hashed

    def __call__(
        self, 
        in_array: mx.array # [B, in_dim]
    ):
        """### MultiHashEncoding.forward
        ###### in `mlx_nerf/encoding/multi_hash.py`

        
        """

        # NOTE: we apply per-level scaling `self.scaled_res` to `in_array`
        in_array_scaled = in_array[..., None, :] * self.scaled_res[..., None] # [B, L, in_dim]

        # NOTE: and rounding it up and down
        in_sc = mx.ceil(in_array_scaled).astype(mx.int32) # in_scaled_ceil
        in_sf = mx.floor(in_array_scaled).astype(mx.int32) # in_scaled_floor
        
        # NOTE: now calculate hashed function for each grid vertices
        X, Y, Z = slice(0, 1), slice(1, 2), slice(2, 3)
        def __grid(x: mx.array, y: mx.array, z: mx.array):
            return mx.concatenate([x[..., X], y[..., Y], z[..., Z]], axis=-1) # [B, L, 3]
        # NOTE: order follows https://subscription.packtpub.com/book/game-development/9781849512824/4/ch04lvl1sec09/indexing-primitives
        # NOTE: `grid_0 == in_sc` and `grid_6 == in_sf`
        grid_0 = __grid(in_sc, in_sc, in_sc)
        grid_1 = __grid(in_sc, in_sf, in_sc)
        grid_2 = __grid(in_sf, in_sf, in_sc)
        grid_3 = __grid(in_sf, in_sc, in_sc)
        grid_4 = __grid(in_sc, in_sc, in_sf)
        grid_5 = __grid(in_sc, in_sf, in_sf)
        grid_6 = __grid(in_sf, in_sf, in_sf)
        grid_7 = __grid(in_sf, in_sc, in_sf)

        # NOTE: hash multigrid points # [B, L, n_features_per_level]
        hashed_0 = self.hash_table(self.hash(grid_0))
        hashed_1 = self.hash_table(self.hash(grid_1))
        hashed_2 = self.hash_table(self.hash(grid_2))
        hashed_3 = self.hash_table(self.hash(grid_3))
        hashed_4 = self.hash_table(self.hash(grid_4))
        hashed_5 = self.hash_table(self.hash(grid_5))
        hashed_6 = self.hash_table(self.hash(grid_6))
        hashed_7 = self.hash_table(self.hash(grid_7))

        # TODO: trilinear interpolation using grid points
        offset = in_array_scaled - in_sf # NOTE: delta(in_array_scaled, in_sf)
        hashed_03 = hashed_0 * offset[..., X] + hashed_3 * (1 - offset[..., X])
        hashed_12 = hashed_1 * offset[..., X] + hashed_2 * (1 - offset[..., X])
        hashed_56 = hashed_5 * offset[..., X] + hashed_6 * (1 - offset[..., X])
        hashed_47 = hashed_4 * offset[..., X] + hashed_7 * (1 - offset[..., X])
        hashed_0312 = hashed_03 * offset[..., Y] + hashed_12 * (1 - offset[..., Y])
        hashed_4756 = hashed_47 * offset[..., Y] + hashed_56 * (1 - offset[..., Y])
        hashed_trilinear_interpolated = hashed_0312 * offset[..., Z] + hashed_4756 * (
            1 - offset[..., Z]
        )  # [B, L, n_features_per_level]

        return (out_encoded := mx.flatten(
            hashed_trilinear_interpolated, 
            start_axis=-2, 
            end_axis=-1,
        )) # [B, L * n_features_per_level]