from pathlib import Path

def import_project_root() -> None:

    return

def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent

