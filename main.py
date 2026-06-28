from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import easyocr
import torch
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from backend.core import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models into RAM...")

    app.state.yolo_front = YOLO(str(config.YOLO_FRONT_MODEL_PATH))
    app.state.yolo_back = YOLO(str(config.YOLO_BACK_MODEL_PATH))

    app.state.ocr_reader = easyocr.Reader(config.EASYOCR_LANGUAGES, gpu=False)

    app.state.face_model = FaceAnalysis(
        name=config.FACE_MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )
    app.state.face_model.prepare(ctx_id=-1, det_size=(640, 640))

    # liveness + tampering — torch.load pattern, adapt based on
    app.state.tampering_model = load_tampering_model(config.TAMPERING_MODEL_PATH)
    app.state.passive_liveness= str(config.PASSIVE_LIVENESS_MODEL_PATH)
    # how passiveliveness.py and predict_baseline.py actually load their models

    print("All models loaded. Server ready.")
    yield
    print("Shutting down...")


app = FastAPI(title="KYC API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}