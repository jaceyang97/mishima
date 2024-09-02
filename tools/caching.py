import os
import pickle
from pathlib import Path
from typing import Any

def cache_data(data: Any, file_path: Path) -> Any:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_cached_data(file_path: Path) -> Any:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None
