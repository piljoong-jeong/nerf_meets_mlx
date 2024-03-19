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
        self.scaled_res = mx.floor(self.min_res * (self.growing_factor ** levels))

        # NOTE: `T` in Table. (1)
        self.hash_table_size = 2 ** self.log2_hashmap_size

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

        



        raise NotImplementedError