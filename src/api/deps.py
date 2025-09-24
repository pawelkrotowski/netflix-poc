from dataclasses import dataclass, field
from typing import Dict, Set, Optional
import numpy as np
import pandas as pd
from fastapi import HTTPException

from .config import (
    ART, DATA, RATINGS_CSV,
    MODEL_PATH, MODEL_CANDIDATES, USERS_MAP_CSV, ITEMS_MAP_CSV, POP_CSV, FEEDBACK_PATH
)
from .domain.model_store import load_als_model_npz, detect_swapped, get_items_matrix
from .domain.popularity import recompute_pop_scores
from .domain.feedback import load_feedback_cache

@dataclass
class AppState:
    model: Optional[object] = None
    swapped: bool = False
    users_map: Optional[pd.DataFrame] = None
    items_map: Optional[pd.DataFrame] = None
    pop_scores: Optional[np.ndarray] = None
    seen_by_user: Dict[int, Set[int]] = field(default_factory=dict)
    bans_by_user: Dict[int, Set[int]] = field(default_factory=dict)
    likes_by_user: Dict[int, Set[int]] = field(default_factory=dict)

STATE = AppState()

def _load_seen(users_map: pd.DataFrame, items_map: pd.DataFrame):
    """Buduje 'seen' z ratingów >=4."""
    out: Dict[int, Set[int]] = {}
    if not RATINGS_CSV.exists():
        return out
    r = pd.read_csv(RATINGS_CSV, usecols=["userId","movieId","rating"])
    r = r[r["rating"] >= 4.0]
    if r.empty:
        return out
    df = (r.merge(users_map, on="userId", how="inner")
            .merge(items_map[["item_index","movieId"]], on="movieId", how="inner"))
    for u, g in df.groupby("user_index"):
        out[int(u)] = set(int(x) for x in g["item_index"].tolist())
    return out

def choose_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    for cand in MODEL_CANDIDATES:
        if cand.exists():
            return cand
    raise RuntimeError(f"No model found. Tried: {MODEL_PATH} and {MODEL_CANDIDATES}")

# ---- Public API for app/routers ----
def init_state():
    # model
    mp = choose_model_path()
    STATE.model = load_als_model_npz(mp)
    STATE.swapped = detect_swapped(STATE.model)
    # maps
    if not USERS_MAP_CSV.exists() or not ITEMS_MAP_CSV.exists():
        raise RuntimeError("Missing users_map.csv / items_map.csv in artifacts")
    STATE.users_map = pd.read_csv(USERS_MAP_CSV)
    STATE.items_map = pd.read_csv(ITEMS_MAP_CSV)
    # popularity
    n_items_eff = get_items_matrix(STATE.model, STATE.swapped).shape[0]
    STATE.pop_scores = recompute_pop_scores(POP_CSV, n_items_eff)
    # seen
    STATE.seen_by_user = _load_seen(STATE.users_map, STATE.items_map)
    # feedback cache
    STATE.bans_by_user, STATE.likes_by_user = load_feedback_cache(FEEDBACK_PATH, STATE.users_map)

def reload_model(path: Optional[str] = None):
    from pathlib import Path
    p = Path(path) if path else choose_model_path()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Model .npz not found")
    STATE.model = load_als_model_npz(p)
    STATE.swapped = detect_swapped(STATE.model)
    n_items_eff = get_items_matrix(STATE.model, STATE.swapped).shape[0]
    STATE.pop_scores = recompute_pop_scores(POP_CSV, n_items_eff)
    return {"model_path": str(p), "swapped": STATE.swapped, "items": int(n_items_eff)}

def reload_artifacts():
    if not USERS_MAP_CSV.exists() or not ITEMS_MAP_CSV.exists():
        raise HTTPException(status_code=404, detail="users_map.csv / items_map.csv not found")
    STATE.users_map = pd.read_csv(USERS_MAP_CSV)
    STATE.items_map = pd.read_csv(ITEMS_MAP_CSV)
    n_items_eff = get_items_matrix(STATE.model, STATE.swapped).shape[0] if STATE.model else int(STATE.items_map["item_index"].max()+1)
    STATE.pop_scores = recompute_pop_scores(POP_CSV, n_items_eff)
    STATE.seen_by_user = _load_seen(STATE.users_map, STATE.items_map)
    return {"users": int(STATE.users_map["user_index"].max()+1), "items": int(n_items_eff)}

def reload_feedback():
    STATE.bans_by_user, STATE.likes_by_user = load_feedback_cache(FEEDBACK_PATH, STATE.users_map)
    return {
        "users_with_bans": len(STATE.bans_by_user),
        "users_with_likes": len(STATE.likes_by_user),
        "total_bans": sum(len(s) for s in STATE.bans_by_user.values()),
        "total_likes": sum(len(s) for s in STATE.likes_by_user.values()),
    }
