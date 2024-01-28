import time
from pathlib import Path
from typing import List

import numpy as onp
import viser
import viser.extras
import viser.transforms as tf
from tqdm.auto import tqdm

from this_project import get_project_root, PJ_PINK

def init_gui(server: viser.ViserServer, **config) -> None:

    num_frames = config.get("num_frames", 1)
    fps = config.get("fps", 1)

    with server.add_gui_folder("Playback"):
        gui_timestep = server.add_gui_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.add_gui_button("Next Frame", disabled=True)
        gui_prev_frame = server.add_gui_button("Prev Frame", disabled=True)
        gui_playing = server.add_gui_checkbox("Playing", True)
        gui_framerate = server.add_gui_slider(
            "FPS", min=1, max=60, step=1, initial_value=fps
        )

    server.configure_theme(
        # titlebar_content="NeRF using MLX", # FIXME: this results blank page
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=PJ_PINK
    )

    return



def main(
    data_path: Path = get_project_root() / "assets/record3d_dance",
    downsample_factor: int = 4,
    max_frames: int = 100,
    share: bool = False,
):

    server = viser.ViserServer()

    init_gui(
        server, 
    )

    while(True):

        time.sleep(0.1)