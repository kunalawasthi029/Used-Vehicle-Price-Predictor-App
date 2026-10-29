"""
utils.py - Utility functions for the Vehicle Price Prediction project.

Contains helper functions for saving/loading models, encoders,
and generating evaluation metrics.
"""

import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------------
# Paths
# -----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_PATH = os.path.join(PROJECT_ROOT, "cleaned_vehicles1.csv")


# -----------------------------------------------
# Model persistence
# -----------------------------------------------
def save_artifact(obj, filename, directory=None):
    """Save a Python object (model, encoder, etc.) to disk using joblib.

    Args:
        obj: The Python object to serialize.
        filename: Name of the output file.
        directory: Target directory. Defaults to MODELS_DIR.

    Returns:
        str: Full path of the saved file.
    """
    if directory is None:
        directory = MODELS_DIR
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    joblib.dump(obj, filepath)
    print(f"  [SAVED] {filepath}")
    return filepath


def load_artifact(filename, directory=None):
    """Load a Python object from disk using joblib.

    Args:
        filename: Name of the file to load.
        directory: Source directory. Defaults to MODELS_DIR.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if directory is None:
        directory = MODELS_DIR
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    return joblib.load(filepath)


# -----------------------------------------------
# Evaluation
# -----------------------------------------------
def evaluate_model(y_true, y_pred, model_name="Model"):
    """Compute regression metrics and print a summary.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.
        model_name: Label used in the printed summary.

    Returns:
        dict: Keys 'model', 'mae', 'rmse', 'r2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'-' * 45}")
    print(f"  {model_name} Performance")
    print(f"{'-' * 45}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")
    print(f"{'-' * 45}")

    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2}


def format_price(price):
    """Format a numeric price as a USD string."""
    return f"${price:,.2f}"
