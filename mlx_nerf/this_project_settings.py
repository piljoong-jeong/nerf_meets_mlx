import sys
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def import_project_root() -> None:

    sys.path.append(str(get_project_root()))

    return

@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    NOTE: each depth of `parent` property represents:
    
        1. removes extension of this file path
        2. parent of this file path
    """
    return Path(__file__).absolute().parent.parent

