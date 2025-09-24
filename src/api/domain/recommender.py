import numpy as np
import pandas as pd
from .model_store import get_items_matrix, get_user_vec

def recommend_for_user_index(state, user_index: int, k: int = 10) -> list[int]:
    """Czysty ALS + maskowanie feedbacku. (bez seen)"""
    model, swapped = state.model, state.swapped
    I = get_items_matrix(model, swapped)
    max_users = model.item_factors.shape[0] if swapped else model.user_factors.shape[0]

    # fallback popularności dla userów poza zakresem
    scores = I @ get_user_vec(model, swapped, min(user_index, max_users-1)) if user_index < max_users else np.linalg.norm(I, axis=1)

    # maskuj feedback
    bans = state.bans_by_user.get(int(user_index), set())
    likes = state.likes_by_user.get(int(user_index), set())
    if bans:
        bs = [i for i in bans if i < len(scores)]
        if bs: scores[bs] = -1e12
    if likes:
        ls = [i for i in likes if i < len(scores)]
        if ls: scores[ls] = -1e12

    N = min(k, len(scores))
    top = np.argpartition(-scores, N-1)[:N]
    return top[np.argsort(-scores[top])].tolist()

def recommend_mixed_for_user_index(state, user_index: int, k: int = 10, alpha: float = 0.8) -> list[int]:
    """Miks ALS + popularność + maska seen + feedback."""
    model, swapped = state.model, state.swapped
    I = get_items_matrix(model, swapped)
    n = I.shape[0]
    scores = state.pop_scores.copy() if state.pop_scores is not None else np.zeros(n, dtype=np.float32)

    # ALS część
    max_users = model.item_factors.shape[0] if swapped else model.user_factors.shape[0]
    if user_index < max_users:
        u_vec = get_user_vec(model, swapped, user_index)
        als = I @ u_vec
        amax, amin = float(als.max()), float(als.min())
        if amax > amin:
            als = (als - amin) / (amax - amin)
        scores = alpha * als + (1.0 - alpha) * scores

    # maskuj seen
    seen = state.seen_by_user.get(int(user_index), set())
    if seen:
        ss = [i for i in seen if i < n]
        if ss: scores[ss] = -1e12

    # maskuj feedback
    bans = state.bans_by_user.get(int(user_index), set())
    if bans:
        bs = [i for i in bans if i < n]
        if bs: scores[bs] = -1e12
    likes = state.likes_by_user.get(int(user_index), set())
    if likes:
        ls = [i for i in likes if i < n]
        if ls: scores[ls] = -1e12

    top = np.argpartition(-scores, min(k, n-1))[:k]
    return top[np.argsort(-scores[top])].tolist()

def popular_topk(state, k: int) -> list[int]:
    import numpy as np
    order = np.argsort(-state.pop_scores) if state.pop_scores is not None else np.array([], dtype=int)
    return order[:min(k, len(order))].tolist()

def items_to_payload(items_map: pd.DataFrame, ids: list[int]) -> list[dict]:
    ix = items_map.set_index("item_index")
    out = []
    for i in ids:
        if i in ix.index:
            rec = ix.loc[i]
            out.append({"item_index": int(i), "movieId": int(rec["movieId"]), "title": str(rec.get("title",""))})
        else:
            out.append({"item_index": int(i), "movieId": None, "title": ""})
    return out
