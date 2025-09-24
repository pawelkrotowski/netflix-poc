# scheduler/trainer.py
import os, json, time, pathlib, requests, socket, getpass, hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# BLAS hygiene
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1, "blas")
except Exception:
    pass

from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
import mlflow

ART = pathlib.Path("/workspace/artifacts")
DATA = pathlib.Path("/workspace/data/raw/ml-latest-small")
ART.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

RATINGS_CSV = DATA / "ratings.csv"
USERS_MAP   = ART / "users_map.csv"
ITEMS_MAP   = ART / "items_map.csv"
POPULAR_CSV = ART / "popular_items.csv"
FEEDBACK    = ART / "feedback.jsonl"
BEST_FILE   = ART / "best_map_at_10.json"
MODEL_LATEST= ART / "model_latest.npz"
MODELS_DIR  = ART / "models"
LAST_RUN_ID = ART / "last_run_id"
MODELS_DIR.mkdir(exist_ok=True)


def _load_feedback_df() -> pd.DataFrame:
    rows = []
    if FEEDBACK.exists():
        with FEEDBACK.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["user_id", "item_index", "movieId", "relevant"]
    )


def _build_ui(ratings: pd.DataFrame, users_map: pd.DataFrame, items_map: pd.DataFrame,
              feedback: pd.DataFrame | None, alpha_up: float = 2.0) -> csr_matrix:
    # bazowe pozytywy z ratingów
    pos = ratings[ratings["rating"] >= 4.0][["userId", "movieId"]].copy()
    pos["rating"] = 1.0
    df = (pos.merge(users_map, on="userId", how="inner")
             .merge(items_map[["item_index", "movieId"]], on="movieId", how="inner"))
    u = df["user_index"].to_numpy(np.int32)
    i = df["item_index"].to_numpy(np.int32)
    v = df["rating"].to_numpy(np.float32)

    # --- NORMALIZACJA FEEDBACKU ---
    # Obsługuje oba przypadki:
    #  - user_index lub user_id
    #  - item_index lub movieId
    def normalize_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
        fb = feedback.copy()

        # uidx
        if "user_index" in fb.columns:
            fb["uidx"] = fb["user_index"]
        elif "user_id" in fb.columns:
            fb = fb.merge(users_map.rename(columns={"userId": "user_id"}),
                          on="user_id", how="left")
            fb = fb.rename(columns={"user_index": "uidx"})
        else:
            fb["uidx"] = np.nan

        # iidx
        if "item_index" in fb.columns:
            fb["iidx"] = fb["item_index"]
        elif "movieId" in fb.columns:
            fb = fb.merge(items_map[["movieId", "item_index"]],
                          on="movieId", how="left")
            fb = fb.rename(columns={"item_index": "iidx"})
        else:
            fb["iidx"] = np.nan

        # zachowaj tylko kompletne i rzutuj na int
        keep_cols = ["uidx", "iidx", "relevant"]
        fb = fb[[c for c in keep_cols if c in fb.columns]]
        fb = fb.dropna(subset=["uidx", "iidx"])
        if "relevant" not in fb.columns:
            fb["relevant"] = True  # domyślnie traktuj jako upvote
        fb["uidx"] = fb["uidx"].astype(int)
        fb["iidx"] = fb["iidx"].astype(int)
        return fb

    if feedback is not None and not feedback.empty:
        fb = normalize_feedback(feedback)

        # 👎 ban – usuń pary user–item z macierzy
        bans = set(fb[fb["relevant"] == False][["uidx", "iidx"]]
                   .itertuples(index=False, name=None))
        if bans:
            mask = np.array([(int(uu), int(ii)) not in bans
                             for uu, ii in zip(u, i)], dtype=bool)
            u, i, v = u[mask], i[mask], v[mask]

        # 👍 upvote – dociąż pary user–item
        ups = fb[fb["relevant"] == True][["uidx", "iidx"]].to_numpy()
        if len(ups):
            u = np.concatenate([u, ups[:, 0].astype(np.int32)])
            i = np.concatenate([i, ups[:, 1].astype(np.int32)])
            v = np.concatenate([v, np.full(len(ups), float(alpha_up), dtype=np.float32)])

    # rozmiary na bazie map (spójne z API)
    n_users = int(users_map["user_index"].max()) + 1
    n_items = int(items_map["item_index"].max()) + 1
    return csr_matrix((v, (u, i)), shape=(n_users, n_items), dtype=np.float32)



def _train_eval(UI: csr_matrix, factors=64, reg=0.02, iters=15, k_eval=10, seed=42):
    rows, cols = UI.nonzero()
    idx_all = np.arange(UI.nnz, dtype=np.int64)
    train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=seed)

    def build(indices):
        r, c = rows[indices], cols[indices]
        vv = UI.data[indices]
        return csr_matrix((vv, (r, c)), shape=UI.shape, dtype=np.float32)

    UI_train, UI_test = build(train_idx), build(test_idx)

    # BM25 + ALS on ITEM×USER
    UIw = bm25_weight(UI_train, K1=1.2, B=0.75).astype(np.float32)
    IUw = UIw.T.tocsr()

    model = AlternatingLeastSquares(
        factors=factors, regularization=reg, iterations=iters, random_state=seed
    )
    model.fit(IUw)

    # swap-safe helpers
    n_items_train, n_users_train = UI_train.shape[1], UI_train.shape[0]
    n_items_model, _ = model.item_factors.shape
    n_users_model, _ = model.user_factors.shape
    SWAPPED = (n_items_model == n_users_train) and (n_users_model == n_items_train)

    def I():
        return model.user_factors if SWAPPED else model.item_factors

    def U(uix):
        return model.item_factors[uix] if SWAPPED else model.user_factors[uix]

    # truth (TEST) + popularity (TRAIN)
    truth = defaultdict(set)
    tr, tc = UI_test.nonzero()
    for r, c in zip(tr, tc):
        truth[r].add(c)

    pop = np.asarray(UI_train.sum(axis=0)).ravel()
    pop_order = np.argsort(-pop)

    def rec(uix, N=10):
        if UI_train.getrow(uix).nnz == 0:
            return pop_order[:N].tolist()
        s = I() @ U(uix)
        seen = [x for x in UI_train.getrow(uix).indices if x < I().shape[0]]
        if seen:
            s[seen] = -1e12
        top = np.argpartition(-s, min(N, I().shape[0] - 1))[:N]
        return top[np.argsort(-s[top])].tolist()

    # evaluation @K
    users = np.array(list(truth.keys()))
    if len(users) > 500:
        rng = np.random.default_rng(seed)
        users = rng.choice(users, size=500, replace=False)

    # clip truth to model items
    clipped_truth = {u: {i for i in items if i < I().shape[0]} for u, items in truth.items()}

    def _eval_with_predictor(predict_fn, k=k_eval):
        precs, recs, maps = [], [], []
        for uix in users:
            t = clipped_truth[uix]
            if not t:
                continue
            p = predict_fn(uix, k)
            inter = len(set(p) & t)
            precs.append(inter / k)
            recs.append(inter / len(t))
            hits = 0
            score = 0.0
            for rank, item in enumerate(p, start=1):
                if item in t:
                    hits += 1
                    score += hits / rank
            maps.append(score / min(k, len(t)))
        return (
            float(np.mean(precs)) if precs else 0.0,
            float(np.mean(recs)) if recs else 0.0,
            float(np.mean(maps)) if maps else 0.0,
        )

    # ALS metrics
    p10, r10, m10 = _eval_with_predictor(lambda uix, k: rec(uix, k), k_eval)

    # Popularity baseline metrics
    def rec_pop(uix, N=10):
        return pop_order[:N].tolist()

    p10_pop, r10_pop, m10_pop = _eval_with_predictor(lambda uix, k: rec_pop(uix, k), k_eval)

    metrics = {
        "precision_at_10": p10,
        "recall_at_10": r10,
        "map_at_10": m10,
        "precision_at_10_pop": p10_pop,
        "recall_at_10_pop": r10_pop,
        "map_at_10_pop": m10_pop,
        "swapped_detected": bool(SWAPPED),
    }
    return model, metrics, UI_train, UI_test


def _save_popularity(ratings: pd.DataFrame, items_map: pd.DataFrame):
    pop = ratings[ratings["rating"] >= 4].groupby("movieId").size().rename("count").reset_index()
    pop = items_map.merge(pop, on="movieId", how="left").fillna({"count": 0})
    pop = pop.sort_values("count", ascending=False).reset_index(drop=True)
    pop.to_csv(POPULAR_CSV, index=False)


def _md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_once():
    assert RATINGS_CSV.exists(), f"Missing {RATINGS_CSV}"
    ratings = pd.read_csv(RATINGS_CSV)

    # maps (must exist; produce them from training notebook first run)
    assert USERS_MAP.exists() and ITEMS_MAP.exists(), "Run training notebook first to produce users_map/items_map."
    users_map = pd.read_csv(USERS_MAP)
    items_map = pd.read_csv(ITEMS_MAP)

    feedback = _load_feedback_df()
    UI = _build_ui(ratings, users_map, items_map, feedback, alpha_up=float(os.getenv("ALS_ALPHA_UP", "2.0")))

    factors = int(os.getenv("ALS_FACTORS", "64"))
    reg     = float(os.getenv("ALS_REG", "0.02"))
    iters   = int(os.getenv("ALS_ITERS", "15"))
    k_eval  = 10

    t0 = time.time()
    model, metrics, UI_train, UI_test = _train_eval(UI, factors, reg, iters, k_eval=k_eval)
    train_eval_sec = time.time() - t0

    # >>> dodatkowe metryki/statystyki
    fb_up = int((feedback["relevant"] == True).sum()) if not feedback.empty else 0
    fb_down = int((feedback["relevant"] == False).sum()) if not feedback.empty else 0
    fb_total = fb_up + fb_down
    model_size_mb = (model.user_factors.nbytes + model.item_factors.nbytes) / 1e6

    data_stats = {
        "n_users": int(UI.shape[0]),
        "n_items": int(UI.shape[1]),
        "nnz_total": int(UI.nnz),
        "nnz_train": int(UI_train.nnz),
        "nnz_test": int(UI_test.nnz),
    }

    # zapis modelu i popularności
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"als_{ts}.npz"
    np.savez_compressed(model_path, user_factors=model.user_factors, item_factors=model.item_factors)
    _save_popularity(ratings, items_map)

    # MLflow
    mlflow.set_experiment("netflix-poc")
    with mlflow.start_run(run_name=f"AUTO_als_{ts}") as run:
        # params
        mlflow.log_param("factors", factors)
        mlflow.log_param("regularization", reg)
        mlflow.log_param("iterations", iters)
        mlflow.log_param("alpha_upvote", float(os.getenv("ALS_ALPHA_UP", "2.0")))
        mlflow.log_param("swapped_detected", metrics["swapped_detected"])
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("user", getpass.getuser())
        try:
            mlflow.set_tag("dataset_md5", _md5(RATINGS_CSV))
        except Exception:
            pass

        # metrics – data stats
        mlflow.log_metric("n_users", data_stats["n_users"])
        mlflow.log_metric("n_items", data_stats["n_items"])
        mlflow.log_metric("nnz_total", data_stats["nnz_total"])
        mlflow.log_metric("nnz_train", data_stats["nnz_train"])
        mlflow.log_metric("nnz_test", data_stats["nnz_test"])

        # metrics – training & model
        mlflow.log_metric("train_eval_sec", float(train_eval_sec))
        mlflow.log_metric("model_size_mb", float(model_size_mb))

        # metrics – feedback
        mlflow.log_metric("feedback_up", fb_up)
        mlflow.log_metric("feedback_down", fb_down)
        mlflow.log_metric("feedback_total", fb_total)

        # metrics – quality (ALS)
        mlflow.log_metric("precision_at_10", metrics["precision_at_10"])
        mlflow.log_metric("recall_at_10", metrics["recall_at_10"])
        mlflow.log_metric("map_at_10", metrics["map_at_10"])

        # metrics – baseline popularity
        mlflow.log_metric("precision_at_10_pop", metrics["precision_at_10_pop"])
        mlflow.log_metric("recall_at_10_pop", metrics["recall_at_10_pop"])
        mlflow.log_metric("map_at_10_pop", metrics["map_at_10_pop"])

        # artifacts
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(USERS_MAP))
        mlflow.log_artifact(str(ITEMS_MAP))
        mlflow.log_artifact(str(POPULAR_CSV))

        # zapisz run_id (do późniejszych dopisków metryk operacyjnych)
        try:
            LAST_RUN_ID.write_text(run.info.run_id)
        except Exception:
            pass

    # deploy only if better
    min_improve = float(os.getenv("MIN_IMPROVE", "0.0001"))
    best_prev = 0.0
    if BEST_FILE.exists():
        try:
            best_prev = json.loads(BEST_FILE.read_text()).get("map_at_10", 0.0)
        except Exception:
            pass

    deployed = False
    if metrics["map_at_10"] >= best_prev + min_improve:
        MODEL_LATEST.write_bytes(model_path.read_bytes())
        BEST_FILE.write_text(json.dumps({"map_at_10": metrics["map_at_10"], "ts": ts}, indent=2))
        deployed = True
        # hot-reload API
        api_reload = os.getenv("API_RELOAD_URL", "http://api:8080/admin/reload-model")
        try:
            requests.post(api_reload, timeout=5)
        except Exception:
            pass
        # optional: reload artifacts (popular_items/seen)
        api_reload_art = os.getenv("API_RELOAD_ART_URL", "http://api:8080/admin/reload-artifacts")
        try:
            requests.post(api_reload_art, timeout=5)
        except Exception:
            pass

    return {
        "metrics": metrics
        | {
            "model_size_mb": float(model_size_mb),
            "feedback_up": fb_up,
            "feedback_down": fb_down,
            "feedback_total": fb_total,
            "train_eval_sec": float(train_eval_sec),
            **data_stats,
        },
        "model_path": str(model_path),
        "deployed": deployed,
    }
