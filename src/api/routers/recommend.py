from fastapi import APIRouter, Query
from ..deps import STATE
from ..domain.recommender import recommend_for_user_index, recommend_mixed_for_user_index, items_to_payload, popular_topk
from ..domain.schemas import RecommendResponse
from ..config import DEFAULT_ALPHA_MIX

router = APIRouter(prefix="/recommend", tags=["recommend"])

@router.get("/user/{user_id}", response_model=RecommendResponse)
def recommend_by_user_id(user_id: int, k: int = Query(10, ge=1, le=50)):
    row = STATE.users_map.loc[STATE.users_map["userId"] == user_id]
    if row.empty:
        ids = popular_topk(STATE, k)  # fallback
        return RecommendResponse(user_id=user_id, k=k, items=items_to_payload(STATE.items_map, ids))
    user_index = int(row["user_index"].iloc[0])
    ids = recommend_for_user_index(STATE, user_index, k)
    return RecommendResponse(user_id=user_id, k=k, items=items_to_payload(STATE.items_map, ids))

@router.get("/index/{user_index}", response_model=RecommendResponse)
def recommend_by_user_index(user_index: int, k: int = Query(10, ge=1, le=50)):
    ids = recommend_for_user_index(STATE, user_index, k)
    return RecommendResponse(user_index=user_index, k=k, items=items_to_payload(STATE.items_map, ids))

@router.get("/mixed/{user_id}", response_model=RecommendResponse)
def recommend_mixed_by_user_id(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    alpha: float = Query(DEFAULT_ALPHA_MIX, ge=0.0, le=1.0)
):
    row = STATE.users_map.loc[STATE.users_map["userId"] == user_id]
    user_index = int(row["user_index"].iloc[0]) if not row.empty else 0
    ids = recommend_mixed_for_user_index(STATE, user_index, k, alpha)
    return RecommendResponse(user_id=user_id, k=k, alpha=alpha, items=items_to_payload(STATE.items_map, ids))
