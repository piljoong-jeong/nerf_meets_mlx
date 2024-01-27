import time

import viser
import viser.extras

def main():

    server = viser.ViserServer()

    num_frames=30

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
            "FPS", min=1, max=60, step=0.1, initial_value=num_frames # loader.fps
        )
        gui_framerate_options = server.add_gui_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    while(True):

        time.sleep(1.0 / num_frames)