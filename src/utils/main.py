# src/api/main.py
import os, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ==== Ścieżki artefaktów / danych ====
ART = Path("/workspace/artifacts")
MODEL_PATH = ART / "als_model.npz"            # startowy model (fallback)
USERS_MAP_CSV = ART / "users_map.csv"
ITEMS_MAP_CSV = ART / "items_map.csv"
POP_CSV = ART / "popular_items.csv"

DATA = Path("/workspace/data/raw/ml-latest-small")
RATINGS_CSV = DATA / "ratings.csv"

# Kandydaci dla admin reload (najpierw spróbuje model_latest.npz)
MODEL_CANDIDATES = [ART / "model_latest.npz", ART / "als_model.npz"]

# ==== Globalny stan ====
app = FastAPI(title="Netflix POC – Recommender API")

model = None
users_map: pd.DataFrame | None = None
items_map: pd.DataFrame | None = None
SWAPPED = False
seen_by_user: dict[int, set[int]] = {}  # user_index -> set(item_index)
pop_df: pd.DataFrame | None = None
pop_scores: np.ndarray | None = None    # długość = n_items_eff

# Feedback cache (natychmiastowe maskowanie w wynikach)
bans_by_user: dict[int, set[int]] = defaultdict(set)   # 👎 – wykluczamy z wyników
likes_by_user: dict[int, set[int]] = defaultdict(set)  # 👍 – też wykluczamy (żeby nie powtarzać)

# ==== Helpery: model, swap, popularność, seen, feedback ====
def load_als_model_npz(model_path: Path):
    from implicit.als import AlternatingLeastSquares
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(f"Pusty lub brak pliku modelu: {model_path}")
    data = np.load(model_path, allow_pickle=False)
    need = {"user_factors", "item_factors"}
    if not need.issubset(set(data.files)):
        raise RuntimeError(f"Plik {model_path} nie zawiera wymaganych kluczy {need}. Zawiera: {set(data.files)}")
    m = AlternatingLeastSquares()
    m.user_factors = data["user_factors"]
    m.item_factors = data["item_factors"]
    return m

def detect_swapped(m) -> bool:
    n_items, _ = m.item_factors.shape
    n_users, _ = m.user_factors.shape
    # Heurystyka: MovieLens zwykle ma więcej itemów niż userów
    return n_items < n_users

def get_user_vec(u: int) -> np.ndarray:
    return model.item_factors[u] if SWAPPED else model.user_factors[u]

def get_items_matrix() -> np.ndarray:
    return model.user_factors if SWAPPED else model.item_factors

def _recompute_pop_scores(n_items_eff: int) -> np.ndarray:
    """Zbuduj/odśwież wektor popularności (min-max do 0..1)."""
    scores = np.zeros(n_items_eff, dtype=np.float32)
    if POP_CSV.exists():
        df = pd.read_csv(POP_CSV)
        if {"item_index", "count"}.issubset(df.columns):
            ix = df["item_index"].to_numpy()
            cnt = df["count"].astype("float32").to_numpy()
            mask = (ix >= 0) & (ix < n_items_eff)
            scores[ix[mask]] = cnt[mask]
    # min-max normalizacja
    mx, mn = float(scores.max()), float(scores.min())
    if mx > mn:
        scores = (scores - mn) / (mx - mn)
    return scores

def build_seen(_users_map: pd.DataFrame, _items_map: pd.DataFrame) -> dict[int, set[int]]:
    """Zbierz 'seen' dla userów na podstawie ratingów >= 4."""
    out: dict[int, set[int]] = {}
    if not RATINGS_CSV.exists():
        return out
    r = pd.read_csv(RATINGS_CSV, usecols=["userId", "movieId", "rating"])
    r = r[r["rating"] >= 4.0]
    if r.empty:
        return out
    df = (r.merge(_users_map, on="userId", how="inner")
            .merge(_items_map[["item_index", "movieId"]], on="movieId", how="inner"))
    for u, g in df.groupby("user_index"):
        out[int(u)] = set(int(x) for x in g["item_index"].tolist())
    return out

FEEDBACK_PATH = ART / "feedback.jsonl"

def _load_feedback_cache():
    """Zbuduj cache banów/lajków z artifacts/feedback.jsonl."""
    bans_by_user.clear()
    likes_by_user.clear()
    if not FEEDBACK_PATH.exists():
        return
    with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uidx = obj.get("user_index")
            if uidx is None and "user_id" in obj and users_map is not None:
                try:
                    row = users_map.loc[users_map["userId"] == int(obj["user_id"])]
                    if not row.empty:
                        uidx = int(row["user_index"].iloc[0])
                except Exception:
                    uidx = None
            iidx = obj.get("item_index")
            rel  = obj.get("relevant")
            if uidx is None or iidx is None:
                continue
            if rel is True:
                likes_by_user[int(uidx)].add(int(iidx))
            elif rel is False:
                bans_by_user[int(uidx)].add(int(iidx))

# ==== Rekomendacje ====
def recommend_for_user_index(user_index: int, k: int = 10) -> List[int]:
    """Czysty ALS z fallbackiem na popularność. Maskuje feedback (bans/likes)."""
    I = get_items_matrix()
    max_users = model.item_factors.shape[0] if SWAPPED else model.user_factors.shape[0]

    # User poza zakresem modelu → popularne
    if user_index >= max_users:
        scores = np.linalg.norm(I, axis=1)
        # maskuj feedback
        bans = bans_by_user.get(int(user_index), set())
        likes = likes_by_user.get(int(user_index), set())
        if bans:
            bs = [i for i in bans if i < len(scores)]
            if bs: scores[bs] = -1e12
        if likes:
            ls = [i for i in likes if i < len(scores)]
            if ls: scores[ls] = -1e12
        N = min(k, len(scores))
        top = np.argpartition(-scores, N-1)[:N]
        return top[np.argsort(-scores[top])].tolist()

    u_vec = get_user_vec(user_index)
    scores = I @ u_vec  # (n_items,)

    # maskuj feedback (bans/likes)
    bans = bans_by_user.get(int(user_index), set())
    likes = likes_by_user.get(int(user_index), set())
    if bans:
        bs = [i for i in bans if i < len(scores)]
        if bs: scores[bs] = -1e12
    if likes:
        ls = [i for i in likes if i < len(scores)]
        if ls: scores[ls] = -1e12

    N = min(k, len(scores))
    top = np.argpartition(-scores, N-1)[:N]
    top = top[np.argsort(-scores[top])]
    return top.tolist()

def popular_topk(k:int) -> list[int]:
    order = np.argsort(-pop_scores) if pop_scores is not None else np.array([], dtype=int)
    return order[:min(k, len(order))].tolist()

def recommend_mixed_for_user_index(user_index: int, k: int = 10, alpha: float = 0.8) -> list[int]:
    """
    alpha ∈ [0,1]: 1.0 = czysty ALS, 0.0 = czysta popularność.
    Maskuje 'seen' oraz feedback (bans/likes).
    """
    I = get_items_matrix()                 # (n_items, factors)
    n = I.shape[0]

    # bazowy score = popularność
    scores = pop_scores.copy() if pop_scores is not None else np.zeros(n, dtype=np.float32)

    # jeśli mamy wektor usera → dodaj ALS
    max_users = model.item_factors.shape[0] if SWAPPED else model.user_factors.shape[0]
    if user_index < max_users:
        u_vec = get_user_vec(user_index)
        als = I @ u_vec  # (n_items,)
        # min-max → 0..1 (żeby mieszać z popularnością)
        amax, amin = float(als.max()), float(als.min())
        if amax > amin:
            als = (als - amin) / (amax - amin)
        scores = alpha * als + (1.0 - alpha) * scores

    # maskuj obejrzane
    seen = seen_by_user.get(int(user_index), set())
    if seen:
        ss = [i for i in seen if i < n]
        if ss:
            scores[ss] = -1e12

    # maskuj feedback (bans/likes)
    bans = bans_by_user.get(int(user_index), set())
    if bans:
        bs = [i for i in bans if i < n]
        if bs: scores[bs] = -1e12
    likes = likes_by_user.get(int(user_index), set())
    if likes:
        ls = [i for i in likes if i < n]
        if ls: scores[ls] = -1e12

    top = np.argpartition(-scores, min(k, n-1))[:k]
    top = top[np.argsort(-scores[top])]
    return top.tolist()

# ==== Startup ====
@app.on_event("startup")
def _load_artifacts():
    global model, users_map, items_map, SWAPPED, pop_df, pop_scores, seen_by_user

    # Model
    candidate = MODEL_PATH if MODEL_PATH.exists() else None
    if candidate is None:
        for cand in MODEL_CANDIDATES:
            if cand.exists():
                candidate = cand
                break
    if candidate is None:
        raise RuntimeError(f"Brak modelu: {MODEL_PATH} lub {MODEL_CANDIDATES}")

    if not USERS_MAP_CSV.exists() or not ITEMS_MAP_CSV.exists():
        raise RuntimeError("Brak mapowań users_map.csv / items_map.csv w /workspace/artifacts")

    model = load_als_model_npz(candidate)
    SWAPPED = detect_swapped(model)

    users_map = pd.read_csv(USERS_MAP_CSV)      # kolumny: user_index, userId
    items_map = pd.read_csv(ITEMS_MAP_CSV)      # kolumny: item_index, movieId, title

    # Popularność
    n_items_eff = get_items_matrix().shape[0]
    pop_df = pd.read_csv(POP_CSV) if POP_CSV.exists() else items_map.assign(count=0)
    pop_scores = _recompute_pop_scores(n_items_eff)

    # Seen
    seen_by_user = build_seen(users_map, items_map)

    # Feedback cache (bany/lajki) – NATYCHMIASTOWY WPŁYW
    _load_feedback_cache()

# ==== Health ====
@app.get("/health")
def health():
    return {
        "status": "ok",
        "items": int(get_items_matrix().shape[0]) if model is not None else 0,
        "users": int(model.item_factors.shape[0] if SWAPPED else model.user_factors.shape[0]) if model is not None else 0,
        "swapped": bool(SWAPPED),
        "seen_users": int(len(seen_by_user)),
        "pop_scores_nonzero": int(int((pop_scores > 0).sum()) if pop_scores is not None else 0),
        "users_with_bans": int(len(bans_by_user)),
        "users_with_likes": int(len(likes_by_user)),
    }

# ==== Publiczne endpointy rekomendacji ====
@app.get("/recommend/user/{user_id}")
def recommend_by_user_id(user_id: int, k: int = Query(10, ge=1, le=50)):
    # mapuj z zewnętrznego userId -> user_index
    row = users_map.loc[users_map["userId"] == user_id]
    if row.empty:
        # user nieznany → popularne ogólnie (lub ALS user_index=0)
        ids = popular_topk(k) if pop_scores is not None else recommend_for_user_index(0, k)
    else:
        user_index = int(row["user_index"].iloc[0])
        ids = recommend_for_user_index(user_index, k)

    # mapuj item_index -> (movieId, title)
    ix = items_map.set_index("item_index")
    items = []
    for i in ids:
        if i in ix.index:
            rec = ix.loc[i]
            items.append({"item_index": int(i), "movieId": int(rec["movieId"]), "title": str(rec.get("title", ""))})
        else:
            items.append({"item_index": int(i), "movieId": None, "title": ""})

    return {"user_id": int(user_id), "k": int(k), "items": items}

@app.get("/recommend/index/{user_index}")
def recommend_by_user_index(user_index: int, k: int = Query(10, ge=1, le=50)):
    ids = recommend_for_user_index(user_index, k)
    ix = items_map.set_index("item_index")
    items = []
    for i in ids:
        if i in ix.index:
            rec = ix.loc[i]
            items.append({"item_index": int(i), "movieId": int(rec["movieId"]), "title": str(rec.get("title", ""))})
        else:
            items.append({"item_index": int(i), "movieId": None, "title": ""})
    return {"user_index": int(user_index), "k": int(k), "items": items}

@app.get("/recommend/mixed/{user_id}")
def recommend_mixed_by_user_id(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.8, ge=0.0, le=1.0)
):
    row = users_map.loc[users_map["userId"] == user_id]
    user_index = int(row["user_index"].iloc[0]) if not row.empty else 0
    ids = recommend_mixed_for_user_index(user_index, k=k, alpha=alpha)

    ix = items_map.set_index("item_index")
    out = []
    for i in ids:
        if i in ix.index:
            rec = ix.loc[i]
            out.append({"item_index": int(i), "movieId": int(rec["movieId"]), "title": str(rec.get("title",""))})
        else:
            out.append({"item_index": int(i), "movieId": None, "title": ""})
    return {"user_id": int(user_id), "k": int(k), "alpha": float(alpha), "items": out}

# ==== Feedback (JSONL) ====
@app.post("/feedback")
def feedback(payload: dict = Body(...)):
    """
    JSON:
    {
      "user_id": 1,        # external userId (lub user_index – jeśli tak, to dodaj to pole)
      "item_index": 123,
      "movieId": 50,       # opcjonalnie
      "relevant": true,    # True / False
      "source": "ui",      # np. "ui" albo "api-test"
      "notes": "optional"
    }
    """
    payload = dict(payload)
    payload["ts"] = datetime.utcnow().isoformat() + "Z"

    # Ustal user_index (jeśli podano tylko user_id)
    uidx = payload.get("user_index")
    if uidx is None and "user_id" in payload and users_map is not None:
        try:
            row = users_map.loc[users_map["userId"] == int(payload["user_id"])]
            if not row.empty:
                uidx = int(row["user_index"].iloc[0])
        except Exception:
            uidx = None
    payload["user_index"] = uidx

    # Uzupełnij tytuł po item_index (jeśli mamy mapę)
    try:
        ix = items_map.set_index("item_index")
        row = ix.loc[payload.get("item_index")]
        payload.setdefault("movieId", int(row["movieId"]))
        payload.setdefault("title", str(row.get("title", "")))
    except Exception:
        pass

    # Zapis do pliku
    with (ART / "feedback.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # NATYCHMIASTOWA aktualizacja cache + seen
    try:
        iidx = int(payload.get("item_index")) if payload.get("item_index") is not None else None
        rel  = payload.get("relevant")
        if uidx is not None and iidx is not None:
            if rel is True:
                likes_by_user[int(uidx)].add(int(iidx))
            elif rel is False:
                bans_by_user[int(uidx)].add(int(iidx))
            # traktuj klik jako „widziane” – unikamy powtórzeń
            seen_by_user.setdefault(int(uidx), set()).add(int(iidx))
    except Exception:
        pass

    return {"ok": True}

# ==== ADMIN: reload model / artifacts / feedback ====
@app.post("/admin/reload-model")
def admin_reload_model(path: Optional[str] = Body(None, embed=True)):
    """
    Przeładuj model z podanej ścieżki (np. /workspace/artifacts/model_latest.npz),
    albo z pierwszego istniejącego kandydata: model_latest.npz / als_model.npz.
    """
    global model, SWAPPED, pop_scores

    # wybierz ścieżkę modelu
    p = Path(path) if path else None
    if p is None:
        for cand in MODEL_CANDIDATES:
            if cand.exists():
                p = cand
                break
    if p is None or not p.exists():
        raise HTTPException(status_code=404, detail="Model .npz not found")

    m = load_als_model_npz(p)
    model = m
    SWAPPED = detect_swapped(model)

    # dopasuj popularność do rozmiaru itemów w modelu
    n_items_eff = get_items_matrix().shape[0]
    pop_scores = _recompute_pop_scores(n_items_eff)

    return {
        "ok": True,
        "model_path": str(p),
        "swapped": bool(SWAPPED),
        "items": int(n_items_eff),
        "factors": int(get_items_matrix().shape[1]),
    }

@app.post("/admin/reload-artifacts")
def admin_reload_artifacts():
    """
    Przeładuj mapy users/items, przelicz 'seen' i popularność (bez zmiany modelu).
    """
    global users_map, items_map, seen_by_user, pop_scores, pop_df

    if not USERS_MAP_CSV.exists() or not ITEMS_MAP_CSV.exists():
        raise HTTPException(status_code=404, detail="users_map.csv / items_map.csv not found")

    users_map = pd.read_csv(USERS_MAP_CSV)
    items_map = pd.read_csv(ITEMS_MAP_CSV)
    pop_df = pd.read_csv(POP_CSV) if POP_CSV.exists() else items_map.assign(count=0)

    seen_by_user = build_seen(users_map, items_map)

    n_items_eff = int(get_items_matrix().shape[0]) if model is not None else int(items_map["item_index"].max() + 1)
    pop_scores = _recompute_pop_scores(n_items_eff)

    return {
        "ok": True,
        "users": int(users_map["user_index"].max() + 1 if "user_index" in users_map else len(users_map)),
        "items": int(n_items_eff),
        "seen_users": int(len(seen_by_user)),
        "pop_scores_nonzero": int(int((pop_scores > 0).sum()) if pop_scores is not None else 0),
    }

@app.post("/admin/reload-feedback")
def admin_reload_feedback():
    """Przeładuj cache feedbacku (bans/likes) z pliku JSONL."""
    _load_feedback_cache()
    n_bans = sum(len(s) for s in bans_by_user.values())
    n_likes = sum(len(s) for s in likes_by_user.values())
    return {
        "ok": True,
        "users_with_bans": len(bans_by_user),
        "users_with_likes": len(likes_by_user),
        "total_bans": n_bans,
        "total_likes": n_likes,
    }
