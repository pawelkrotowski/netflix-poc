# Netflix POC – README

Ten projekt to lekki POC systemu rekomendacji „w stylu Netflixa” oparty o:

- **MovieLens (ml-latest-small)** jako dane wejściowe
- **implicit ALS** (rekomendacje na macierzy implicit user–item)
- **MLflow** do śledzenia eksperymentów i artefaktów
- **FastAPI** jako warstwa serwisowa (REST)
- **Streamlit** jako prosty interfejs użytkownika
- **(Opcjonalnie)** proces **cyklicznego trenowania** z wykorzystaniem feedbacku od użytkowników

Projekt działa na **CPU** i **GPU** (jeśli dostępne). GPU nie jest wymagane.

---

## Spis treści
1. [Architektura](#architektura)
2. [Wymagania wstępne](#wymagania-wstępne)
3. [Szybki start](#szybki-start)
4. [Struktura katalogów](#struktura-katalogów)
5. [Serwisy w docker-compose](#serwisy-w-docker-compose)
6. [Notebooki – przepływ pracy](#notebooki--przepływ-pracy)
7. [API – endpoints](#api--endpoints)
8. [UI – Streamlit](#ui--streamlit)
9. [Trenowanie cykliczne](#trenowanie-cykliczne)
10. [MLflow – metryki i artefakty](#mlflow--metryki-i-artefakty)
11. [Feedback użytkownika](#feedback-użytkownika)
12. [Cold‑start i miks ALS+Popularność](#coldstart-i-miks-alspopularność)
13. [Konfiguracja i zmienne środowiskowe](#konfiguracja-i-zmienne-środowiskowe)
14. [Rozwiązywanie problemów](#rozwiązywanie-problemów)
15. [Wkład i konwencje](#wkład-i-konwencje)

---

## Architektura
```
MovieLens → Notebook (ALS) → Artefakty (model, mapy) → API (FastAPI) → UI (Streamlit)
                                          ↘ MLflow (metryki/parametry/artefakty)
                           ↘ Scheduler (cykliczny retrain z feedbackiem)
```
Kluczowe artefakty w `./artifacts/`:
- `als_model.npz` lub `model_latest.npz` — wektory `user_factors` i `item_factors`
- `users_map.csv` — mapowanie `user_index ↔ userId`
- `items_map.csv` — mapowanie `item_index ↔ movieId` + `title`
- `popular_items.csv` — ranking popularności (do miksu i cold‑start)
- `run_config.json` — zapis parametrów bieżącego runu

---

## Wymagania wstępne
- **Ubuntu 24.04** (inne Linuxy też ok)
- **Docker** i **Docker Compose**
- (GPU, opcjonalnie) sterownik **NVIDIA** + **NVIDIA Container Toolkit**

> Jeśli nie masz GPU lub narzędzi NVIDIA – nic nie szkodzi. POC działa na CPU.

---

## Szybki start
1. Skopiuj repo i przygotuj `.env` (przykładowe wartości):
   ```env
   API_PORT=8080
   UI_PORT=8501
   MLFLOW_PORT=5001
   ```
2. Podnieś bazowe serwisy (Postgres, MLflow, Jupyter):
   ```bash
   docker compose up -d postgres mlflow jupyter
   docker compose logs -f jupyter
   ```
   Jupyter: `http://localhost:8888`

3. W Jupyterze odpal notebook **`01_clean_train_mlflow_swap_safe.ipynb`** (Restart & Run All).
   Notebook: pobierze dane, wytrenuje ALS, zapisze artefakty do `./artifacts/` i zaloguje metryki do **MLflow**.

4. Podnieś API i UI:
   ```bash
   docker compose --profile later up -d api ui
   docker compose logs -f api
   docker compose logs -f ui
   ```
   - API docs: `http://localhost:${API_PORT}/docs`
   - UI: `http://localhost:${UI_PORT}`

5. (Opcjonalnie) Włącz cykliczny trening:
   ```bash
   docker compose --profile later up -d trainer
   docker compose logs -f trainer
   ```

---

## Struktura katalogów
```
├─ artifacts/                # modele, mapy, configi, popularność, feedback
│  ├─ als_model.npz
│  ├─ model_latest.npz       # model, który ładuje API (opcjonalny symlink/kopia)
│  ├─ users_map.csv
│  ├─ items_map.csv
│  ├─ popular_items.csv
│  ├─ feedback.jsonl         # zapisy 👍👎 z UI
│  └─ run_config.json
├─ data/
│  └─ raw/ml-latest-small/   # dane MovieLens (automatycznie pobierane w notebooku)
├─ notebooks/
│  ├─ 01_clean_train_mlflow_swap_safe.ipynb
│  └─ (opcjonalnie) 04_tuning_als.ipynb
├─ scheduler/
│  ├─ trainer.py             # jednorazowy train + log + deploy lepszego modelu
│  └─ train_loop.py          # pętla cykliczna
├─ src/
│  └─ api/main.py            # FastAPI (REST)
├─ ui/
│  └─ app.py                 # Streamlit (frontend)
├─ docker/
│  └─ jupyter.Dockerfile     # obraz z python + data science + implicit
├─ docker-compose.yml
└─ .env
```

---

## Serwisy w docker-compose
- **postgres**: metastore MLflow
- **mlflow**: UI eksperymentów (`http://localhost:${MLFLOW_PORT}`)
- **jupyter**: środowisko notebooków (`http://localhost:8888`)
- **api**: FastAPI (`http://localhost:${API_PORT}`)
- **ui**: Streamlit (`http://localhost:${UI_PORT}`)
- **trainer** *(profil `later`)*: cykliczne trenowanie i deploy lepszych modeli

> Serwisy `api`, `ui`, `trainer` startują zwykle z profilem `later`, aby odciążyć start. Używaj `--profile later` przy podnoszeniu.

---

## Notebooki – przepływ pracy
### 01_clean_train_mlflow_swap_safe.ipynb (zalecany)
- **Import danych** MovieLens (pobiera automatycznie, jeśli brak)
- Budowa macierzy **implicit** (rating ≥ 4.0)
- Split **train/test** po interakcjach
- Ważenie **BM25** na TRAIN
- Trening **ALS** na **ITEM×USER**
- **Eval**: Precision@10, Recall@10, MAP@10 (manualnie, z maskowaniem „seen”)
- **Swap‑safe**: automatyczne wykrywanie ewentualnej zamiany wymiarów w `implicit` (rzadki bug) i odporne rekomendacje
- Zapis artefaktów i log do **MLflow**

### (Opcjonalnie) 04_tuning_als.ipynb
- Prosty random/grid search nad `factors`, `regularization`, `iterations`
- Log runów do MLflow, wybór „best” po `map_at_10`
- Kopia najlepszego modelu do `artifacts/model_latest.npz`

> **Wskazówka**: zawsze używaj **Restart & Run All**, aby uniknąć resztek stanu kernela.

---

## API – endpoints
Po starcie API zobacz **Swagger UI**: `http://localhost:${API_PORT}/docs`

- `GET /health` — status, liczba użytkowników i itemów, flaga `swapped`
- `GET /recommend/user/{user_id}?k=10` — TOP‑K rekomendacji ALS (maskuje „seen”)
- `GET /recommend/mixed/{user_id}?k=10&alpha=0.8` — miks ALS + popularność (`alpha`= udział ALS)
- `GET /similar/movie/{movie_id}?k=10` — podobne tytuły do danego filmu
- `POST /feedback` — zapis interakcji z UI/klienta (👍/👎), ląduje w `artifacts/feedback.jsonl`
- `POST /admin/reload-model` — przeładuj model z `artifacts/model_latest.npz` (hot‑reload)
- (opcjonalnie) `GET /debug/model-shapes` — kształty macierzy modelu (pomoc w debugowaniu)

**Przykłady:**
```bash
# Zdrowie
curl -s http://localhost:8080/health | jq .

# Rekomendacje ALS
curl -s "http://localhost:8080/recommend/user/1?k=10" | jq .

# MIXED (ALS+Popularność)
curl -s "http://localhost:8080/recommend/mixed/1?k=10&alpha=0.7" | jq .

# Podobne do filmu (movieId=1)
curl -s "http://localhost:8080/similar/movie/1?k=10" | jq .

# Feedback (przykład 👍)
curl -s -X POST http://localhost:8080/feedback \
  -H 'Content-Type: application/json' \
  -d '{"user_id":1, "item_index":123, "movieId":50, "relevant":true, "source":"api-test"}' | jq .

# Hot‑reload modelu
curl -s -X POST http://localhost:8080/admin/reload-model | jq .
```

---

## UI – Streamlit
- Start: `docker compose --profile later up -d ui`
- UI: `http://localhost:${UI_PORT}`
- Funkcje: wybór `userId`, tryb **ALS** lub **MIXED**, suwak `alpha`, przyciski feedbacku 👍/👎, przycisk „🔁 Podobne”.

> Uwaga: UI w kontenerze rozmawia z API pod adresem `http://api:8080`. Zmienisz to przez `API_URL` w środowisku serwisu UI.

---

## Trenowanie cykliczne
Serwis **trainer** (profil `later`) uruchamia okresowo:
- budowę macierzy UI na podstawie danych + feedbacku (👍 zwiększa wagę, 👎 filtruje parę user–item)
- trening ALS
- ewaluację (MAP@10)
- zapis nowego modelu i `popular_items.csv`
- log do **MLflow**
- **deploy tylko lepszego** modelu (aktualizacja `model_latest.npz`) i **hot‑reload** API

Konfigurujesz przez zmienne środowiskowe (patrz niżej): `TRAIN_INTERVAL_SEC`, `ALS_FACTORS`, `ALS_REG`, `ALS_ITERS`, `MIN_IMPROVE`.

Start:
```bash
docker compose --profile later up -d trainer
```

---

## MLflow – metryki i artefakty
- UI: `http://localhost:${MLFLOW_PORT}`
- Run’y zapisują: parametry (`factors`, `regularization`, `iterations`, `k_eval`, `swapped_detected`), metryki (`precision_at_10`, `recall_at_10`, `map_at_10`), artefakty (model, mapy, popularność, config).

> W notebooku „swap‑safe” nazwa runu to **`ALS_swap_safe`**.

---

## Feedback użytkownika
- UI wysyła feedback na `/feedback` (JSONL w `artifacts/feedback.jsonl`)
- Proces trenowania wykorzystuje feedback:
  - **👍**: waga interakcji ↑ (np. `+2.0`)
  - **👎**: para user–item jest **filtrowana** (ban) przy budowie macierzy
- Dzięki temu kolejne treningi stopniowo adaptują się do sygnałów użytkowników.

---

## Cold‑start i miks ALS+Popularność
- `popular_items.csv` trzymany w `artifacts/` (generowany w notebooku i/lub przez trainer)
- Endpoint `/recommend/mixed/{user_id}` zwraca miks **ALS (alpha)** i **popularności (1−alpha)**
- Dla nowych użytkowników (brak „seen”) miks daje sensowniejsze wyniki niż czysty ALS.

---

## Konfiguracja i zmienne środowiskowe
`.env` (przykład):
```env
API_PORT=8080
UI_PORT=8501
MLFLOW_PORT=5001
```
Serwis **ui**:
```yaml
environment:
  UI_PORT: ${UI_PORT}
  API_URL: http://api:8080
```
Serwis **trainer**:
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
> Dodatkowo w notebookach zalecamy: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

---

## Rozwiązywanie problemów
**1) BLAS / wielowątkowość / niskie osiągi**
- Ustaw: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` (na starcie notebooka/serwisu)
- (Opcj.) `threadpoolctl.threadpool_limits(1, "blas")`

**2) `implicit` – zamiana wymiarów (swap) po treningu**
- Używaj notebooka **swap‑safe**: wykrywa `SWAPPED` i liczy rekomendacje/metryki poprawnie

**3) `connection refused` z UI do API**
- W UI używaj `API_URL=http://api:8080` (wewnątrz sieci Compose `localhost` wskazuje na kontener UI)

**4) API nie wstaje – `model.item_factors is None`**
- Różnice wersji `implicit` a format `.npz`. W API używamy bezpiecznego loadera przez `numpy.load` i ręczne osadzenie faktorów

**5) GPU / `--gpus all`**
- GPU jest opcjonalne. Jeśli chcesz używać GPU, zainstaluj sterowniki NVIDIA i NVIDIA Container Toolkit, a w compose uruchom kontener z dostępem do GPU. W razie braku GPU – wszystko działa na CPU.

**6) Porty zajęte**
- Zmień `API_PORT`, `UI_PORT`, `MLFLOW_PORT` w `.env` i podnieś ponownie serwisy.

**7) „items mismatch”/asercje w starych notebookach**
- Korzystaj z wersji **swap‑safe** i usuń asercje; działaj przez helpery `get_items_matrix()`/`get_user_vec()`

---

## Wkład i konwencje
- PR’y mile widziane – pamiętaj o krótkim opisie zmian i testach manualnych
- Styl: PEP8 dla Pythona, czyste notebooki (bez zbędnych komórek)
- Każdy większy eksperyment: **log do MLflow** + zapis artefaktów
- Przy deployu nowego modelu: kopiuj do `artifacts/model_latest.npz` i wywołuj `/admin/reload-model`

---

**Pytania?**
- Wejdź do MLflow, sprawdź ostatnie run’y i artefakty
- Uderz w `/health` lub `/debug/model-shapes` po stronie API
- Zajrzyj do logów: `docker compose logs -f <serwis>`

