# NOTE: we might import modules through iteration, 
# NOTE: but intellisense might not able to recognize those modules for autosuggestion
# NOTE: hence, do import manually

from .__viser_record3d import main as viser_record3d
from .__viser_image_learning import main as viser_image_learning
from .__viser_gui import main as viser_gui
from .nerf import main as nerf

# TODO: set common theme here
