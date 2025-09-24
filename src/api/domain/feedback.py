# src/api/domain/feedback.py
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Optional, Tuple

import pandas as pd

def load_feedback_cache(
    feedback_path: Path,
    users_map: Optional[pd.DataFrame]
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Zwraca (bans_by_user, likes_by_user).
    - akceptuje feedback.jsonl z polami: user_index / user_id, item_index / movieId, relevant
    - mapuje user_id -> user_index przez users_map (jeśli potrzeba)
    """
    bans: Dict[int, Set[int]] = defaultdict(set)
    likes: Dict[int, Set[int]] = defaultdict(set)

    if not feedback_path.exists():
        return bans, likes

    with feedback_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # user_index
            uidx = obj.get("user_index")
            if uidx is None and "user_id" in obj and users_map is not None:
                try:
                    row = users_map.loc[users_map["userId"] == int(obj["user_id"])]
                    if not row.empty:
                        uidx = int(row["user_index"].iloc[0])
                except Exception:
                    uidx = None

            # item_index
            iidx = obj.get("item_index")
            if iidx is None and "movieId" in obj:
                # jeżeli chcesz wspierać movieId -> item_index, podaj items_map tutaj
                # aktualny moduł nie ma dostępu do items_map — zostawiamy tylko item_index
                pass

            rel = obj.get("relevant")
            if uidx is None or iidx is None:
                continue

            if rel is True:
                likes[int(uidx)].add(int(iidx))
            elif rel is False:
                bans[int(uidx)].add(int(iidx))

    return bans, likes
