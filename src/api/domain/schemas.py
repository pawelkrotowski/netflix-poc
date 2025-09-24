from pydantic import BaseModel, Field
from typing import Optional, List

class RecItem(BaseModel):
    item_index: int
    movieId: Optional[int] = None
    title: str = ""

class RecommendResponse(BaseModel):
    user_id: Optional[int] = None
    user_index: Optional[int] = None
    k: int
    alpha: Optional[float] = None
    items: List[RecItem]

class FeedbackIn(BaseModel):
    user_id: Optional[int] = None
    user_index: Optional[int] = None
    item_index: Optional[int] = None
    movieId: Optional[int] = None
    relevant: bool = Field(..., description="True=like, False=dislike")
    source: Optional[str] = "ui"
    notes: Optional[str] = None
