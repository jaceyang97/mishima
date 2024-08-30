import pickle
import os


def cache_data(data, filename):
    """Saves data to a file using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_cached_data(filename):
    """Loads data from a pickle file if it exists."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None
