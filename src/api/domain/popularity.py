import numpy as np
import pandas as pd
from pathlib import Path

def recompute_pop_scores(pop_csv: Path, n_items_eff: int) -> np.ndarray:
    scores = np.zeros(n_items_eff, dtype=np.float32)
    if pop_csv.exists():
        df = pd.read_csv(pop_csv)
        if {"item_index","count"}.issubset(df.columns):
            ix = df["item_index"].to_numpy()
            cnt = df["count"].astype("float32").to_numpy()
            mask = (ix >= 0) & (ix < n_items_eff)
            scores[ix[mask]] = cnt[mask]
    # min-max → 0..1
    mx, mn = float(scores.max()), float(scores.min())
    if mx > mn:
        scores = (scores - mn) / (mx - mn)
    return scores
