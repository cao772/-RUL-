from __future__ import annotations

import csv
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import FEATURE_NAMES, PredictionEngine

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"

app = FastAPI(title="光伏组件健康状态与RUL预测平台", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def engine() -> PredictionEngine:
    return PredictionEngine(ROOT)


class SampleRequest(BaseModel):
    sample_index: int = Field(ge=0)


class FeatureRequest(BaseModel):
    features: list[list[float]]


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    e = engine()
    return {"ok": True, "model_loaded": True, "device": str(e.device), "samples": e.n_samples}


@app.get("/api/meta")
def meta() -> dict:
    return engine().meta()


@app.get("/api/sample/{sample_index}")
def sample(sample_index: int) -> dict:
    try:
        return engine().sample_payload(sample_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict/sample")
def predict_sample(req: SampleRequest) -> dict:
    try:
        return engine().predict_sample(req.sample_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict/features")
def predict_features(req: FeatureRequest) -> dict:
    try:
        return engine().predict_features(np.asarray(req.features, dtype=np.float32))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="请上传 CSV 文件")
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 必须为 UTF-8 编码") from exc

    reader = csv.DictReader(io.StringIO(text))
    missing = [name for name in FEATURE_NAMES if name not in (reader.fieldnames or [])]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV 缺少字段: {', '.join(missing)}")

    rows = []
    try:
        for row in reader:
            rows.append([float(row[name]) for name in FEATURE_NAMES])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="CSV 存在无法转换为数字的字段") from exc
    if len(rows) != 60:
        raise HTTPException(status_code=400, detail=f"CSV 必须正好 60 行数据，当前 {len(rows)} 行")

    try:
        return engine().predict_features(np.asarray(rows, dtype=np.float32))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
