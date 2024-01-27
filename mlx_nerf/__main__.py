"""


"""

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tyro

from mlx_nerf import entrypoints


if __name__ == "__main__":
    tyro.cli(entrypoints.viser_record3d)