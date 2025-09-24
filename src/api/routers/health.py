from fastapi import APIRouter
from ..deps import STATE
from ..domain.model_store import get_items_matrix

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    if STATE.model is None:
        return {"status": "init"}
    return {
        "status": "ok",
        "items": int(get_items_matrix(STATE.model, STATE.swapped).shape[0]),
        "users": int(STATE.model.item_factors.shape[0] if STATE.swapped else STATE.model.user_factors.shape[0]),
        "swapped": bool(STATE.swapped),
        "seen_users": int(len(STATE.seen_by_user)),
        "users_with_bans": int(len(STATE.bans_by_user)),
        "users_with_likes": int(len(STATE.likes_by_user)),
        "pop_scores_nonzero": int(int((STATE.pop_scores > 0).sum()) if STATE.pop_scores is not None else 0),
    }
