import json
import pprint
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

import imageio.v2 as imageio
import numpy as onp
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import viser
import viser.extras

from this_project import get_project_root, PJ_PINK
from mlx_nerf import config_parser
from mlx_nerf.dataset.dataloader import load_blender_data, validate_dataset
from mlx_nerf.models.NeRF import create_NeRF

def main(
    path_dataset: Path = Path.home() / "Downloads" / "NeRF",
    max_iter: int = 10000,
    
):
    
    parser = config_parser.config_parser()
    args = parser.parse_args(args=[])
    args.config = "configs/lego.txt"
    path_config = path_dataset / args.config
    
    configs = config_parser.load_config(None, path_config)
    pprint.pprint(configs)

    dir_dataset = configs["datadir"]
    images, poses, render_poses, hwf, i_split = load_blender_data(path_dataset / dir_dataset)
    # validate_dataset(path_dataset / dir_dataset)

    render_kwargs_train, render_kwargs_test, idx_iter, optimizer = create_NeRF(args)

