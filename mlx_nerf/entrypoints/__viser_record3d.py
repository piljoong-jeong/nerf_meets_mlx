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
        # gui_framerate_options = server.add_gui_button_group(
        #     "FPS options", ("10", "20", "30", "60")
        # )

    

    return



def main(
    data_path: Path = get_project_root() / "assets/record3d_dance",
    downsample_factor: int = 4,
    max_frames: int = 100,
    share: bool = False,
):
    print("Loading frames!")
    loader = viser.extras.Record3dLoader(data_path) # TODO: implement own
    num_frames = min(max_frames, loader.num_frames())

    server = viser.ViserServer()
    if share:
        server.request_share_url()

    init_gui(
        server, 
        num_frames=num_frames, fps=loader.fps
    )

    # Load in frames.
    server.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: List[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        frame = loader.get_frame(i)
        position, color = frame.get_point_cloud(downsample_factor)

        # Add base frame.
        frame_nodes.append(server.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position,
            colors=color,
            point_size=0.01,
            point_shape="rounded",
        )

        # Place the frustum.
        fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
        aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
        server.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frame.rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
        )

        # Add some axes.
        server.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == 0


    server.configure_theme(
        # titlebar_content="NeRF using MLX", # FIXME: this results blank page
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=PJ_PINK
    )

    while(True):

        time.sleep(1.0 / num_frames)