import os
import pprint
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
# import tk
# from tkinter import filedialog # FIXME: not working in MacOS
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

class GUIItem:
    btn_select_image: Optional[viser.GuiButtonHandle] = None
    cbox_learning: Optional[viser.GuiButtonHandle] = None
    is_learning = False
    slider_iter: Optional[viser.GuiInputHandle] = None
    idx_iter = 0
    pass

gui_items = GUIItem()

def select_image(path_default: Union[str, Path]) -> None:

    

    return

def toggle_learning() -> None:
    
    gui_items.is_learning = not gui_items.is_learning
    print(f"{gui_items.is_learning=}")

    if not gui_items.is_learning:
        gui_items.slider_iter.value = 0

    return
    

def main(
    path_assets: Path = get_project_root() / "assets",
    downsample_factor: int = 4,
    max_frames: int = 10000,
    share: bool = False,
):

    server = viser.ViserServer()
    server.configure_theme(
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=PJ_PINK
    )

    # gui_items.btn_select_image = server.add_gui_button("Select Image"); gui_items.btn_select_image.on_click(lambda _: select_image(path_assets))

    gui_items.cbox_learning = server.add_gui_checkbox("Learning?", False)
    gui_items.cbox_learning.on_update(lambda _: toggle_learning())
    gui_items.slider_iter = server.add_gui_slider(
        "# Iterations",
        min=0,
        max=max_frames,
        step=1,
        initial_value=0,
        disabled=True,
    )

    
    
    while True:
        
        
        if gui_items.slider_iter.value >= max_frames:
            continue

        if gui_items.is_learning:
            gui_items.slider_iter.value += 1
        
        
        time.sleep(1.0 / 60)
        pass