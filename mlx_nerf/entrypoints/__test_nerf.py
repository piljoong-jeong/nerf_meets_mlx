import os
from functools import partial
from pathlib import Path

import imageio.v2
import numpy as onp
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
import viser
import viser.extras
from tqdm import trange

from this_project import get_project_root, PJ_PINK
from mlx_nerf import config_parser
from mlx_nerf.dataset.dataloader import load_blender_data
from mlx_nerf.models.NeRF import create_NeRF
from mlx_nerf.rendering import ray, render
from mlx_nerf.rendering.render import render_rays, raw2outputs
from mlx_nerf import sampling


def main(
    path_dataset: Path = Path.home() / "Downloads" / "NeRF",
    max_iter: int = 5000,
    
):
    
    parser = config_parser.config_parser()
    args = parser.parse_args(args=[])
    args.config = "configs/lego.txt"
    path_config = path_dataset / args.config
    
    configs = config_parser.load_config(None, path_config)
    args = config_parser.update_NeRF_args(configs)
    
    dir_dataset = configs["datadir"]
    images, poses, render_poses, hwf, i_split = load_blender_data(path_dataset / dir_dataset)

    
    