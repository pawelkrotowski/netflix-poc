from fastapi import FastAPI
from .deps import init_state
from .routers import health, recommend, feedback, admin

app = FastAPI(title="Netflix POC – Recommender API (modular)")

@app.on_event("startup")
def _startup():
    init_state()

# Routery
app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(admin.router)
