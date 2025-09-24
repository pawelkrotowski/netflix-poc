# Netflix POC – README (for developers)

This project is a lightweight **Netflix‑style recommender POC** built with:

- **MovieLens (ml-latest-small)** as the dataset
- **implicit ALS** (matrix factorization for implicit feedback)
- **MLflow** for experiment tracking and artifacts
- **FastAPI** as the service layer (REST)
- **Streamlit** as a simple UI
- **(Optional)** a **scheduled retraining process** that incorporates user feedback

Runs on **CPU** and can use **GPU** if available. GPU is optional.

---

## Table of contents
1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick start](#quick-start)
4. [Project structure](#project-structure)
5. [Services (docker-compose)](#services-docker-compose)
6. [Notebooks & workflow](#notebooks--workflow)
7. [API – endpoints](#api--endpoints)
8. [UI – Streamlit](#ui--streamlit)
9. [Scheduled training](#scheduled-training)
10. [MLflow – metrics & artifacts](#mlflow--metrics--artifacts)
11. [User feedback](#user-feedback)
12. [Cold‑start & ALS+Popularity mix](#cold-start--alspopularity-mix)
13. [Configuration & environment](#configuration--environment)
14. [Troubleshooting](#troubleshooting)
15. [Contributing](#contributing)

---

## Architecture
```
MovieLens → Notebook (ALS) → Artifacts (model, maps) → API (FastAPI) → UI (Streamlit)
                                      ↘ MLflow (metrics/params/artifacts)
                       ↘ Scheduler (periodic retrain with feedback)
```
Key artifacts in `./artifacts/`:
- `als_model.npz` or `model_latest.npz` — `user_factors` & `item_factors`
- `users_map.csv` — `user_index ↔ userId`
- `items_map.csv` — `item_index ↔ movieId` + `title`
- `popular_items.csv` — popularity ranking (used for mixing & cold‑start)
- `run_config.json` — parameters of the current run
- `feedback.jsonl` — UI/API feedback log (👍/👎)

---

## Prerequisites
- **Ubuntu 24.04** (other Linux distros should work)
- **Docker** and **Docker Compose**
- (Optional, for GPU) **NVIDIA driver** + **NVIDIA Container Toolkit**

> No GPU? No problem — the POC runs on CPU.

---

## Quick start
1. Clone the repo and create `.env` (example values):
   ```env
   API_PORT=8080
   UI_PORT=8501
   MLFLOW_PORT=5001
   ```
2. Bring up core services (Postgres, MLflow, Jupyter):
   ```bash
   docker compose up -d postgres mlflow jupyter
   docker compose logs -f jupyter
   ```
   Jupyter: `http://localhost:8888`

3. In Jupyter, run **`01_clean_train_mlflow_swap_safe.ipynb`** (Restart & Run All).
   The notebook downloads data, trains ALS, writes artifacts to `./artifacts/`, and logs metrics to **MLflow**.

4. Start API & UI:
   ```bash
   docker compose --profile later up -d api ui
   docker compose logs -f api
   docker compose logs -f ui
   ```
   - API docs: `http://localhost:${API_PORT}/docs`
   - UI: `http://localhost:${UI_PORT}`

5. (Optional) Enable scheduled retraining:
   ```bash
   docker compose --profile later up -d trainer
   docker compose logs -f trainer
   ```

---

## Project structure
```
├─ artifacts/                # models, maps, config, popularity, feedback
│  ├─ als_model.npz
│  ├─ model_latest.npz       # model loaded by API (copy/symlink)
│  ├─ users_map.csv
│  ├─ items_map.csv
│  ├─ popular_items.csv
│  ├─ feedback.jsonl
│  └─ run_config.json
├─ data/
│  └─ raw/ml-latest-small/   # MovieLens data (auto-downloaded by notebook)
├─ notebooks/
│  ├─ 01_clean_train_mlflow_swap_safe.ipynb
│  └─ (optional) 04_tuning_als.ipynb
├─ scheduler/
│  ├─ trainer.py             # one-shot train + log + deploy-if-better
│  └─ train_loop.py          # periodic loop
├─ src/
│  └─ api/main.py            # FastAPI (REST)
├─ ui/
│  └─ app.py                 # Streamlit (frontend)
├─ docker/
│  └─ jupyter.Dockerfile     # Python + DS + implicit
├─ docker-compose.yml
└─ .env
```

---

## Services (docker-compose)
- **postgres**: MLflow backend store
- **mlflow**: experiment UI (`http://localhost:${MLFLOW_PORT}`)
- **jupyter**: notebook environment (`http://localhost:8888`)
- **api**: FastAPI (`http://localhost:${API_PORT}`)
- **ui**: Streamlit (`http://localhost:${UI_PORT}`)
- **trainer** *(profile `later`)*: periodic training and best‑model deploy

> We keep `api`, `ui`, and `trainer` under the `later` profile to avoid heavy startup. Use `--profile later` when needed.

---

## Notebooks & workflow
### 01_clean_train_mlflow_swap_safe.ipynb (recommended)
- **Data import** (auto downloads MovieLens if missing)
- Build **implicit** matrix (rating ≥ 4.0)
- **Train/test split** by interactions
- **BM25** weighting on TRAIN
- Train **ALS** on **ITEM×USER**
- **Evaluation**: Precision@10, Recall@10, MAP@10 (manual, masks "seen")
- **Swap‑safe**: detects rare item/user swap in `implicit` and evaluates correctly
- Saves artifacts & logs to **MLflow**

### (Optional) 04_tuning_als.ipynb
- Simple random/grid search over `factors`, `regularization`, `iterations`
- Logs runs to MLflow; pick **best** by `map_at_10`
- Copy best model to `artifacts/model_latest.npz`

> **Tip:** Always use **Restart & Run All** to avoid stale kernel state.

---

## API – endpoints
After starting the API, open **Swagger UI**: `http://localhost:${API_PORT}/docs`

- `GET /health` — status, counts, and `swapped` flag
- `GET /recommend/user/{user_id}?k=10` — ALS TOP‑K (masks items already seen)
- `GET /recommend/mixed/{user_id}?k=10&alpha=0.8` — ALS + Popularity mix (`alpha` = ALS weight)
- `GET /similar/movie/{movie_id}?k=10` — titles similar to a given movie
- `POST /feedback` — persist user feedback (👍/👎) into `artifacts/feedback.jsonl`
- `POST /admin/reload-model` — hot‑reload model from `artifacts/model_latest.npz`
- (optional) `GET /debug/model-shapes` — factor matrix shapes (for debugging)

**Examples:**
```bash
# Health
curl -s http://localhost:8080/health | jq .

# ALS recommendations
curl -s "http://localhost:8080/recommend/user/1?k=10" | jq .

# MIXED (ALS+Popularity)
curl -s "http://localhost:8080/recommend/mixed/1?k=10&alpha=0.7" | jq .

# Similar by movieId
curl -s "http://localhost:8080/similar/movie/1?k=10" | jq .

# Feedback (thumbs up)
curl -s -X POST http://localhost:8080/feedback \
  -H 'Content-Type: application/json' \
  -d '{"user_id":1, "item_index":123, "movieId":50, "relevant":true, "source":"api-test"}' | jq .

# Hot‑reload the model
curl -s -X POST http://localhost:8080/admin/reload-model | jq .
```

---

## UI – Streamlit
- Start: `docker compose --profile later up -d ui`
- Open: `http://localhost:${UI_PORT}`
- Features: select `userId`, choose **ALS** or **MIXED**, control `alpha`, send feedback 👍/👎, and open "🔁 Similar" list.

> Note: The UI (in its own container) calls API at `http://api:8080`. You can change it via `API_URL` in the UI service env.

---

## Scheduled training
The **trainer** service (profile `later`) periodically:
- builds the UI matrix from data + feedback (👍 increases weight, 👎 filters the pair)
- trains ALS
- evaluates on MAP@10
- saves the new model and `popular_items.csv`
- logs to **MLflow**
- **deploys only if better** (updates `model_latest.npz`) and **hot‑reloads** the API

Configure via env vars (see below): `TRAIN_INTERVAL_SEC`, `ALS_FACTORS`, `ALS_REG`, `ALS_ITERS`, `MIN_IMPROVE`.

Start:
```bash
docker compose --profile later up -d trainer
```

---

## MLflow – metrics & artifacts
- UI: `http://localhost:${MLFLOW_PORT}`
- Runs log: parameters (`factors`, `regularization`, `iterations`, `k_eval`, `swapped_detected`), metrics (`precision_at_10`, `recall_at_10`, `map_at_10`), artifacts (model, maps, popularity, config).

> In the swap‑safe notebook the run is named **`ALS_swap_safe`**.

---

## User feedback
- UI sends feedback to `/feedback` and appends JSON lines to `artifacts/feedback.jsonl`.
- Retraining uses feedback:
  - **👍**: increases interaction weight (e.g., `+2.0`)
  - **👎**: the user–item pair is **filtered out** when building the matrix

This makes subsequent models adapt to user signals.

---

## Cold‑start & ALS+Popularity mix
- `popular_items.csv` in `artifacts/` (generated by the notebook and/or trainer)
- `/recommend/mixed/{user_id}` blends **ALS (alpha)** and **popularity (1−alpha)**
- For brand‑new users (no history) the mix yields more sensible results than pure ALS.

---

## Configuration & environment
`.env` (example):
```env
API_PORT=8080
UI_PORT=8501
MLFLOW_PORT=5001
```
UI service:
```yaml
environment:
  UI_PORT: ${UI_PORT}
  API_URL: http://api:8080
```
Trainer service:
```yaml
environment:
  MLFLOW_TRACKING_URI: http://mlflow:${MLFLOW_PORT}
  API_RELOAD_URL: http://api:${API_PORT}/admin/reload-model
  TRAIN_INTERVAL_SEC: 86400
  ALS_FACTORS: 64
  ALS_REG: 0.02
  ALS_ITERS: 15
  MIN_IMPROVE: 0.0001
  OPENBLAS_NUM_THREADS: 1
```
> In notebooks, also set: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

---

## Troubleshooting
**1) BLAS / threading / performance**
- Set: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` (in notebooks/services)
- (Optional) `threadpoolctl.threadpool_limits(1, "blas")`

**2) `implicit` item/user **swap** after training**
- Use the **swap‑safe** notebook; it detects `SWAPPED` and evaluates correctly via helper accessors

**3) UI → API "connection refused"**
- Inside Docker network, use `API_URL=http://api:8080` (not `localhost`)

**4) API fails: `model.item_factors is None`**
- Version mismatch / loading issue; API uses a safe loader (`numpy.load`) and assigns factors manually

**5) GPU / `--gpus all`**
- GPU is optional. If needed, install NVIDIA drivers and NVIDIA Container Toolkit and start the container with GPU access. CPU is fully supported.

**6) Ports already in use**
- Change `API_PORT`, `UI_PORT`, `MLFLOW_PORT` in `.env` and restart services

**7) Old notebooks fail on `assert` (items mismatch)**
- Use the **swap‑safe** version and rely on `get_items_matrix()` / `get_user_vec()` instead of asserts

---

## Contributing
- PRs welcome — include a short description and manual test notes
- Style: PEP8 for Python; keep notebooks tidy (no stale cells)
- Log every meaningful experiment to **MLflow** and persist artifacts
- When deploying a new model: copy to `artifacts/model_latest.npz` and call `/admin/reload-model`

---

**Questions?**
- Open MLflow and inspect recent runs & artifacts
- Hit `/health` or `/debug/model-shapes` on the API
- Check logs: `docker compose logs -f <service>`

