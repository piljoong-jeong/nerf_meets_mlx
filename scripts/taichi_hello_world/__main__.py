import os
DIR_THIS_FILE = os.path.dirname(__file__)

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(2*n, n))

@ti.func
def complex_sqr(z):
    return ti.Vector(
        [
            z[0] ** 2 - z[1] ** 2,
            z[1] * z[0] * 2, 
        ]
    )

@ti.kernel
def paint(t: float):
    for i, j in pixels:
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i/n - 1, j/n - 0.5]) * 2

        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


# NOTE: GUI
name_this_function = "Julia Set"
gui = ti.GUI(name_this_function, res=(2*n, n))

import os
import imageio.v2 as imageio
writer = imageio.get_writer(os.path.join(DIR_THIS_FILE, f"{name_this_function}.mp4"), fps=60)
i=0
while gui.running and i < 600:
    paint(i*0.03)
    gui.set_image(pixels)
    gui.show()

    pixels_np = (pixels.to_numpy(dtype=np.float32) * 255.0).astype(np.uint8) # [H, W]
    writer.append_data(pixels_np.transpose())
    i+=1

# NOTE: save video
writer.close()