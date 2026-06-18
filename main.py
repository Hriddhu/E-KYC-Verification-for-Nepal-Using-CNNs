from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.core import config
from config import TAMPERING_MODEL_PATH,YOLO_FRONT_MODEL_PATH,YOLO_BACK_MODEL_PATH,PASSIVE_LIVENESS_MODEL_PATH


# ── Model state (preloaded at startup) ──────────────────
class ModelState:
    pass


# ── Lifespan (startup + shutdown) ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load all models here once
    # store in app.state
    app.state.tampering_model = TAMPERING_MODEL_PATH
    app.state.liveness_model =  PASSIVE_LIVENESS_MODEL_PATH
    app.state.yolo_front_model =  YOLO_FRONT_MODEL_PATH
    app.state.yolo_back_model = YOLO_BACK_MODEL_PATH
    yield
    
    del app.state.face_detector
    del app.state.ocr_reader
    del app.state.face_model
    del app.state.liveness_model
    # cleanup on shutdown


# ── App ─────────────────────────────────────────────────
app = FastAPI(
    title="KYC API",
    lifespan=lifespan
)


# ── Middleware ───────────────────────────────────────────


# ── Health check ─────────────────────────────────────────
@app.get("/health")
def health():
    pass


# ── Routers ──────────────────────────────────────────────
# include routers here