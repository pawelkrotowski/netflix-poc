from fastapi import APIRouter, Body
from ..deps import reload_model, reload_artifacts, reload_feedback

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/reload-model")
def admin_reload_model(path: str | None = Body(None, embed=True)):
    info = reload_model(path)
    return {"ok": True, **info}

@router.post("/reload-artifacts")
def admin_reload_artifacts():
    info = reload_artifacts()
    return {"ok": True, **info}

@router.post("/reload-feedback")
def admin_reload_feedback():
    info = reload_feedback()
    return {"ok": True, **info}
