import os
from pathlib import Path

def get_root_path():
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT_PATH = get_root_path()
NEUROX_PATH = ROOT_PATH / "src" / "clustering" / "NeuroX"