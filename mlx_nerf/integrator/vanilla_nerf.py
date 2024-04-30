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
from mlx_nerf import sampling
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
        self.positional_encoding = SinusoidalEncoding(
            in_dim=3, 
            n_freqs=10, 
            min_freq_exp=0.0, 
            max_freq_exp=9.0, 
            is_include_input=True
        )
        self.directional_encoding = SinusoidalEncoding(
            in_dim=3, 
            n_freqs=4, 
            min_freq_exp=0.0, 
            max_freq_exp=3.0, 
            is_include_input=True
        )

        # NOTE: coarse
        self.sampler_uniform = uniform.sample_z # TODO: partial function to `near` & `far`-agnostic?
        self.model_coarse = NeRF(
            channel_input=self.positional_encoding.get_out_dim(), 
            channel_input_views=self.directional_encoding.get_out_dim(), 
            is_use_view_directions=True
        )
        mx.eval(self.model_coarse.parameters())
        
        # NOTE: fine
        self.sampler_importance = sample_from_inverse_cdf_torch
        self.model_fine = NeRF(
            channel_input=self.positional_encoding.get_out_dim(), 
            channel_input_views=self.directional_encoding.get_out_dim(), 
            is_use_view_directions=True
        )
        mx.eval(self.model_fine.parameters())
        
        # NOTE: renderer


        # NOTE: optimizer
        self.optimizer = optim.Adam(learning_rate=(learning_rate := 5e-4), betas=(0.9, 0.999))

        # FIXME: refactor
        self.near = 2.0
        self.far = 6.0
        self.n_depth_samples = 64
        self.n_importance_samples = 128
        self.perturb = 0.0

    # NOTE: all computations to backpropagate must be fused in `mlx` kernel
    def get_outputs(
        self, 
        rays,   # [2, B, 3]
        target, # [B, 3]
    ):
        
        rays_o, rays_d = rays
        rays_shape = rays_d.shape
        n_rays = rays_shape[0]
        viewdirs = rays_d
        viewdirs = viewdirs / mx.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = mx.reshape(viewdirs, [-1, 3]).astype(mx.float32)

        near = self.near * mx.ones_like(rays_d[..., :1])
        far = self.far * mx.ones_like(rays_d[..., :1])
        
        """
        rays = mx.concatenate(
            [rays_o, rays_d, near, far], # TODO: should `near` and `far` included for every ray tensor?
            axis=-1
        )
        rays_linear = mx.concatenate([rays, viewdirs], axis=-1)
        """

        # NOTE: uniform depth sampling
        z_vals = uniform.sample_z(near, far, self.n_depth_samples)
        z_vals = sampling.add_noise_z(z_vals, self.perturb)

        # NOTE: encode positional samples
        pos = rays_o[..., None, :] + (z_vals[..., :, None] * rays_d[..., None, :]) # NOTE: [B, n_depth_samples, 3]
        embedded_pos = self.positional_encoding(
            # NOTE: `pos_flat`: [B*n_depth_samples, 3]
            (pos_flat := mx.reshape(pos, [-1, pos.shape[-1]]))
        ) # [B*n_depth_samples, self.positional_encoding.get_out_dim()]

        # NOTE: encode directional samples
        dir = mx.repeat(viewdirs[:, None, :], repeats=pos.shape[1], axis=1)
        embedded_dir = self.directional_encoding(
            # NOTE: `dir_flat`: [B*n_depth_samples, 3]
            (dir_flat := mx.reshape(dir, [-1, dir.shape[-1]]))
        ) # [B*n_depth_samples, self.directional_encoding.get_out_dim()]

        embedded = mx.concatenate([embedded_pos, embedded_dir], axis=-1) # [B*n_depth_samples, `embedded_pos.shape[-1]+embedded_dir.shape[-1]`]

        model_outputs_coarse = self.model_coarse.forward(embedded) # [B*n_depth_samples, RGBA]
        # NOTE: reshape [B*n_depth_samples, RGBA] -> [B, n_depth_samples, RGBA]
        outputs = mx.reshape(
            model_outputs_coarse, 
            [n_rays, self.n_depth_samples, model_outputs_coarse.shape[-1]] # TODO: double-check
        )

        rgb_coarse, disp_coarse, acc_coarse, weights, depth_map = raw2outputs(
            outputs, z_vals, rays_d, (raw_noise_std:=0.0), (white_bkgd:=False), (pytest:=False)
        )

        mse_coarse = mx.mean((rgb_coarse - target) ** 2)
        print(f"{mse_coarse=}")

        return NotImplementedError
    
