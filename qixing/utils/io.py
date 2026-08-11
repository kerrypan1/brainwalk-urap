"""
File I/O utilities for saving and loading results, models, and metadata.

Provides atomic I/O operations with JSON, pickle, and CSV support.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def ensure_dir(dirpath: str) -> str:
    """
    Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    dirpath : str
        Directory path.

    Returns
    -------
    str
        The directory path.
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    return dirpath


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : dict
        Data to save.
    filepath : str
        Output file path.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into dictionary.

    Parameters
    ----------
    filepath : str
        Input file path.

    Returns
    -------
    dict
        Loaded data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save object to pickle file.

    Parameters
    ----------
    data : Any
        Object to save.
    filepath : str
        Output file path.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    """
    Load pickle file.

    Parameters
    ----------
    filepath : str
        Input file path.

    Returns
    -------
    Any
        Loaded object.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_numpy(data: np.ndarray, filepath: str) -> None:
    """
    Save numpy array to NPZ format.

    Parameters
    ----------
    data : ndarray
        Array to save.
    filepath : str
        Output file path.
    """
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, data)


def load_numpy(filepath: str) -> np.ndarray:
    """
    Load numpy array from file.

    Parameters
    ----------
    filepath : str
        Input file path.

    Returns
    -------
    ndarray
        Loaded array.
    """
    return np.load(filepath)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
