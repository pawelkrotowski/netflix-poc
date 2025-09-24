from pathlib import Path
from typing import Final
import os

# Katalog artefaktów i dane
ART: Final[Path] = Path("/workspace/artifacts")
DATA: Final[Path] = Path("/workspace/data/raw/ml-latest-small")
RATINGS_CSV: Final[Path] = DATA / "ratings.csv"

# Pliki artefaktów
MODEL_PATH: Final[Path] = ART / "als_model.npz"       # startowy fallback
MODEL_CANDIDATES = [ART / "model_latest.npz", ART / "als_model.npz"]
USERS_MAP_CSV: Final[Path] = ART / "users_map.csv"
ITEMS_MAP_CSV: Final[Path] = ART / "items_map.csv"
POP_CSV: Final[Path]       = ART / "popular_items.csv"
FEEDBACK_PATH: Final[Path] = ART / "feedback.jsonl"

# Tuning
DEFAULT_ALPHA_MIX: float = 0.8

# Env helpers
def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
