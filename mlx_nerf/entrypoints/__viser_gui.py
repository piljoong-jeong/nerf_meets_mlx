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


def del_gui(gui):
    if not None is gui: gui.remove()

class GUIItem:
    btn_select_image: Optional[viser.GuiButtonHandle] = None
    cbox_learning: Optional[viser.GuiButtonHandle] = None
    is_learning = False
    slider_iter: Optional[viser.GuiInputHandle] = None
    idx_iter = 0

    tbox_max_iter: Optional[viser.GuiInputHandle] = None;           max_iter: int = 0
    btn_start_learning: Optional[viser.GuiButtonHandle] = None
    btn_stop_learning: Optional[viser.GuiButtonHandle] = None

    pass

gui_items = GUIItem()

def select_image(path_default: Union[str, Path]) -> None:

    

    return

def toggle_learning() -> None:
    
    gui_items.is_learning = not gui_items.is_learning

    if not gui_items.is_learning:
        gui_items.slider_iter.value = 0

    return
    
def start_learning(server: viser.ViserServer, max_iter: int) -> None:

    del_gui(gui_items.btn_start_learning)
    del_gui(gui_items.tbox_max_iter)
    

    create_gui_stop_learning(server, max_iter=max_iter)

    gui_items.is_learning = True


    return

def create_gui_start_learning(server: viser.ViserServer, **kwargs) -> None:

    max_iter = kwargs.get("max_iter", 10000)

    gui_items.tbox_max_iter = server.add_gui_number(
        "Target Iterations", 
        max_iter
    )
    gui_items.btn_start_learning = server.add_gui_button(
        "Start Learning"
    )
    gui_items.btn_start_learning.on_click(lambda _: start_learning(server, gui_items.tbox_max_iter.value))

    return

def stop_learning(server: viser.ViserServer, **kwargs):

    del_gui(gui_items.slider_iter)
    del_gui(gui_items.btn_stop_learning)

    create_gui_start_learning(server, max_iter=gui_items.max_iter)

    gui_items.is_learning = False

    return

def create_gui_stop_learning(server: viser.ViserServer, **kwargs) -> None:

    max_iter = kwargs.get("max_iter", 300)

    gui_items.slider_iter = server.add_gui_slider(
        "# Iterations", 
        min=0, 
        max=max_iter, 
        step=1, 
        initial_value=0, 
        disabled=True
    )
    gui_items.max_iter = max_iter
    gui_items.btn_stop_learning = server.add_gui_button(
        "Stop Learning"
    )
    gui_items.btn_stop_learning.on_click(lambda _: stop_learning(server))


    return

def main(
    path_assets: Path = get_project_root() / "assets",
    downsample_factor: int = 4,
    max_iter: int = 10000,
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

    gui_items.max_iter = max_iter
    create_gui_start_learning(server, max_iter=gui_items.max_iter)


    while True:
        
        
        if not None is gui_items.slider_iter and gui_items.slider_iter.value < gui_items.max_iter and gui_items.is_learning:
            gui_items.slider_iter.value += 1

        
        
        time.sleep(1.0 / 60)
        pass