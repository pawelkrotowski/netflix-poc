from fastapi import APIRouter, Body
from datetime import datetime
import json
from ..deps import STATE
from ..config import FEEDBACK_PATH
from ..domain.schemas import FeedbackIn

router = APIRouter(tags=["feedback"])

@router.post("/feedback")
def feedback(payload: FeedbackIn = Body(...)):
    data = payload.model_dump()
    data["ts"] = datetime.utcnow().isoformat() + "Z"

    # ustal user_index jeśli podano tylko user_id
    uidx = data.get("user_index")
    if uidx is None and data.get("user_id") is not None:
        row = STATE.users_map.loc[STATE.users_map["userId"] == int(data["user_id"])]
        if not row.empty:
            uidx = int(row["user_index"].iloc[0])
    data["user_index"] = uidx

    # uzupełnij tytuł po item_index (opcjonalnie)
    try:
        ix = STATE.items_map.set_index("item_index")
        row = ix.loc[data.get("item_index")]
        data.setdefault("movieId", int(row["movieId"]))
        data.setdefault("title", str(row.get("title","")))
    except Exception:
        pass

    # dopisz do JSONL
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # natychmiastowa aktualizacja cache + seen
    try:
        iidx = int(data.get("item_index")) if data.get("item_index") is not None else None
        if uidx is not None and iidx is not None:
            if data.get("relevant") is True:
                STATE.likes_by_user.setdefault(int(uidx), set()).add(int(iidx))
            elif data.get("relevant") is False:
                STATE.bans_by_user.setdefault(int(uidx), set()).add(int(iidx))
            STATE.seen_by_user.setdefault(int(uidx), set()).add(int(iidx))
    except Exception:
        pass

    return {"ok": True}
