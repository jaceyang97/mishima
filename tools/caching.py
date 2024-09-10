import pickle
import os
from typing import Any
from pathlib import Path

def cache_data(data: Any, file_path: Path) -> None:
    """Saves data to a file using pickle."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_cached_data(file_path: Path) -> Any:
    """Loads data from a pickle file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None
