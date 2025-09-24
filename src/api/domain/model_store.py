from pathlib import Path
import numpy as np

def load_als_model_npz(model_path: Path):
    from implicit.als import AlternatingLeastSquares
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty model file: {model_path}")
    data = np.load(model_path, allow_pickle=False)
    if not {"user_factors","item_factors"}.issubset(set(data.files)):
        raise RuntimeError(f"Invalid npz keys in {model_path}: {set(data.files)}")
    m = AlternatingLeastSquares()
    m.user_factors = data["user_factors"]
    m.item_factors = data["item_factors"]
    return m

def detect_swapped(model) -> bool:
    n_items, _ = model.item_factors.shape
    n_users, _ = model.user_factors.shape
    return n_items < n_users  # w MovieLens zwykle items > users

def get_items_matrix(model, swapped):
    return model.user_factors if swapped else model.item_factors

def get_user_vec(model, swapped, u):
    return model.item_factors[u] if swapped else model.user_factors[u]
