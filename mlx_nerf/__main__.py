"""


"""


# from utils import get_project_root
import this_project_settings
DIR_PROJECT_ROOT2 = str(this_project_settings.get_project_root())
print(f"{DIR_PROJECT_ROOT2=}")
import sys
sys.path.append(DIR_PROJECT_ROOT2)


import tyro


from mlx_nerf import entrypoints

if __name__ == "__main__":
    tyro.cli(entrypoints.viser_record3d)