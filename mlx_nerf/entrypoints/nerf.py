import pprint
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

import imageio.v3 as imageio
import numpy as onp
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import viser
import viser.extras

from this_project import get_project_root, PJ_PINK
from mlx_nerf import config_parser
from mlx_nerf.dataset.dataloader import load_blender_data

def main(
    dir_dataset: Path = Path.home() / "Downloads" / "NeRF",
    max_iter: int = 10000,
    
):
    path_config = dir_dataset / "configs/lego.txt"
    configs = config_parser.load_config(None, path_config)
    pprint.pprint(configs)

    images, poses, render_poses, hwf, i_split = load_blender_data(dir_dataset / configs["datadir"])
    print(images[0])





