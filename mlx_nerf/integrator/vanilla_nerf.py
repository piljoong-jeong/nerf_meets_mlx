import json
import pprint
import os
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

import imageio.v2
import numpy as onp
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


from this_project import get_project_root
from mlx_nerf.encoding.sinusoidal import SinusoidalEncoding
from mlx_nerf.integrator import Integrator
from mlx_nerf.models.NeRF import NeRF
from mlx_nerf.rendering import ray, render
from mlx_nerf.rendering.render import render_rays, raw2outputs
from mlx_nerf.sampling import sample_from_inverse_cdf_torch, uniform

class VanillaNeRFIntegrator(Integrator):
    def __init__(self, config) -> None:
        super().__init__(config)

        """
        NOTE: code flow

        1. Given ray batches & GTs, 
        2. `self.sampler_uniform` augments ray batches into set of positions
        3. `self.positional_encoding` & `self.directional_encoding` encodes input points & directions into encodings
        4. `self.model_coarse` inputs encoded features, and returns `dict_outputs_coarse`
            * loss calculation & backprop via `self.rgb_renderer`
        5. with 2. & `dict_outputs_coarse[WEIGHT]`, generate importance samples using `self.sampler_importance`
            * IMPORTANT: cast between `mlx.array` and `torch.tensor` should be done before & after calling `self.sampler_importance`
        6. encode importance samples & directions 
        7. `self.model_fine` inputs encoded importance & direction samples, and returns `dict_outputs_fine`
            * loss calculation * backprop via `self.rgb_renderer`
        """

        # NOTE: define encoders, used regardless of coarse|fine network
        positional_encoding = SinusoidalEncoding(
            in_dim=3, 
            n_freqs=10, 
            min_freq_exp=0.0, 
            max_freq_exp=9.0, 
            is_include_input=True
        )
        directional_encoding = SinusoidalEncoding(
            in_dim=3, 
            n_freqs=4, 
            min_freq_exp=0.0, 
            max_freq_exp=3.0, 
            is_include_input=True
        )

        # NOTE: coarse
        self.sampler_uniform = uniform.sample_z # TODO: partial function to `near` & `far`-agnostic?
        self.model_coarse = NeRF(
            channel_input=positional_encoding.get_out_dim(), 
            channel_input_views=directional_encoding.get_out_dim(), 
            is_use_view_directions=True
        )
        
        # NOTE: fine
        self.sampler_importance = sample_from_inverse_cdf_torch()
        self.model_fine = NeRF(
            channel_input=positional_encoding.get_out_dim(), 
            channel_input_views=directional_encoding.get_out_dim(), 
            is_use_view_directions=True
        )
        
        # NOTE: 
