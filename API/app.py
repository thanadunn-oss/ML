"""Skin lesion demo API and a small browser UI.

The model bundle is deliberately external to this code.  To deploy a later
approved model, point MODEL_DIR at its folder containing model.pt and
serving_config.json, then provide CLASS_MAPPING_PATH if it lives elsewhere.
"""
import io
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "artifacts" / "model-v2"))
CLASS_MAPPING_PATH = Path(os.getenv("CLASS_MAPPING_PATH", BASE_DIR / "artifacts" / "model-v2" / "class_mapping.json"))
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "predictions.db"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.70"))
# The baseline has poor melanoma recall; keep all melanoma predictions in review.
ALWAYS_REVIEW_CLASSES = {x.strip() for x in os.getenv("ALWAYS_REVIEW_CLASSES", "melanoma").split(",") if x.strip()}

app = FastAPI(title="Skin Lesion Classifier", version="1.0.0")
model: torch.nn.Module | None = None
model_info: dict[str, Any] = {}
eval_transform: Any = None
load_error: str | None = None


def build_model(arch: str, n_classes: int) -> torch.nn.Module:
    if arch == "resnet18":
        network = models.resnet18(weights=None)
        network.fc = torch.nn.Linear(network.fc.in_features, n_classes)
        return network
    if arch == "efficientnet_b0":
        network = models.efficientnet_b0(weights=None)
        network.classifier[1] = torch.nn.Linear(network.classifier[1].in_features, n_classes)
        return network
    raise ValueError(f"Unsupported architecture: {arch}")


def load_model() -> None:
    """Load a bundle once.  Errors are retained so /ready can report them."""
    global model, model_info, eval_transform, load_error
    try:
        config = json.loads((MODEL_DIR / "serving_config.json").read_text(encoding="utf-8"))
        mapping = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
        classes = config["classes"]
        if classes != mapping["classes"]:
            raise ValueError("Class order in serving_config.json and class_mapping.json differs")
        network = build_model(config["arch"], len(classes))
        # model.pt is supplied by the training team and is treated as a trusted artifact.
        state = torch.load(MODEL_DIR / "model.pt", map_location="cpu")
        network.load_state_dict(state)
        network.eval()
        model = network
        model_info = {
            **config,
            "model_version": MODEL_DIR.name,
            "label_schema_version": mapping.get("label_schema_version", "unknown"),
        }
        eval_transform = transforms.Compose([
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ])
        load_error = None
    except Exception as exc:  # shown via readiness rather than crashing the server
        model = None
        model_info = {}
        eval_transform = None
        load_error = f"{type(exc).__name__}: {exc}"


@contextmanager
def database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialise_database() -> None:
    with database() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                preprocessing_version TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                scores_json TEXT NOT NULL,
                review_required INTEGER NOT NULL,
                processing_ms INTEGER NOT NULL,
                feedback_class TEXT,
                feedback_note TEXT,
                feedback_at TEXT
            )
        """)


@app.on_event("startup")
def startup() -> None:
    initialise_database()
    load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail={"ready": False, "error": load_error})
    return {"ready": True, "model_version": model_info["model_version"], "classes": model_info["classes"]}


def require_ready() -> None:
    if model is None or eval_transform is None:
        raise HTTPException(status_code=503, detail="Model is not ready. Check /ready for details.")


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    require_ready()
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(io.BytesIO(content)) as uploaded:
            rgb_image = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    started = time.perf_counter()
    tensor = eval_transform(rgb_image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor)[0], dim=0).tolist()
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    scores = {label: round(float(score), 6) for label, score in zip(model_info["classes"], probabilities)}
    predicted_class = max(scores, key=scores.get)
    confidence = scores[predicted_class]
    review_required = confidence < REVIEW_THRESHOLD or predicted_class in ALWAYS_REVIEW_CLASSES
    prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
    result = {
        "prediction_id": prediction_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "scores": scores,
        "review_required": review_required,
        "model_version": model_info["model_version"],
        "preprocessing_version": model_info["preprocessing_version"],
        "processing_ms": elapsed_ms,
    }
    with database() as conn:
        conn.execute("""INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)""", (
            prediction_id, datetime.now(timezone.utc).isoformat(), result["model_version"],
            result["preprocessing_version"], predicted_class, confidence, json.dumps(scores),
            int(review_required), elapsed_ms,
        ))
    return result


class Feedback(BaseModel):
    prediction_id: str
    confirmed_class: str
    note: str = Field(default="", max_length=1000)


@app.post("/feedback")
def feedback(payload: Feedback) -> dict[str, str]:
    require_ready()
    if payload.confirmed_class not in model_info["classes"]:
        raise HTTPException(status_code=422, detail={"message": "Unknown class", "allowed_classes": model_info["classes"]})
    with database() as conn:
        changed = conn.execute("""UPDATE predictions SET feedback_class=?, feedback_note=?, feedback_at=? WHERE prediction_id=?""", (
            payload.confirmed_class, payload.note, datetime.now(timezone.utc).isoformat(), payload.prediction_id,
        )).rowcount
    if not changed:
        raise HTTPException(status_code=404, detail="Prediction ID not found.")
    return {"status": "saved", "prediction_id": payload.prediction_id}


@app.get("/reviews")
def reviews() -> list[dict[str, Any]]:
    with database() as conn:
        rows = conn.execute("""SELECT * FROM predictions WHERE review_required=1 ORDER BY created_at DESC LIMIT 100""").fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["review_required"] = bool(item["review_required"])
        item["scores"] = json.loads(item.pop("scores_json"))
        results.append(item)
    return results


@app.get("/", response_class=HTMLResponse)
def web() -> str:
    return WEB_PAGE


WEB_PAGE = r'''<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Skin Lesion Classifier</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 18px;background:#f7f8fb;color:#18212f}.card{background:white;padding:24px;border-radius:14px;box-shadow:0 2px 12px #14213d18;margin-bottom:18px}button{background:#1769aa;color:white;border:0;padding:11px 16px;border-radius:8px;font-weight:650;cursor:pointer}button:disabled{opacity:.6}input,select,textarea{width:100%;box-sizing:border-box;margin:8px 0 14px;padding:9px}img{max-width:100%;max-height:320px;display:none;margin:12px 0;border-radius:9px}.warning{color:#8a4100;background:#fff1df;padding:12px;border-radius:8px}.score{display:grid;grid-template-columns:130px 1fr 64px;gap:8px;align-items:center;margin:7px 0}.bar{height:11px;background:#e7ebf0;border-radius:9px}.bar i{display:block;height:100%;background:#287bb8;border-radius:9px}small{color:#566170}</style></head><body><h1>Skin Lesion Classifier</h1><p><small>เครื่องมือสาธิตเพื่อการคัดกรองเท่านั้น ไม่ใช่คำวินิจฉัยทางการแพทย์</small></p><section class="card"><h2>อัปโหลดภาพ</h2><input id="image" type="file" accept="image/jpeg,image/png,image/webp"><img id="preview" alt="ตัวอย่างภาพ"><button id="predict">วิเคราะห์ภาพ</button><p id="error" class="warning" hidden></p></section><section class="card" id="result" hidden><h2>ผลการวิเคราะห์</h2><p><b id="label"></b> <span id="confidence"></span></p><div id="review"></div><div id="scores"></div><p><small id="meta"></small></p><hr><h3>ยืนยันหรือแก้ไขผล</h3><select id="confirmed"></select><textarea id="note" placeholder="หมายเหตุ (ไม่บังคับ)"></textarea><button id="sendFeedback">บันทึก feedback</button><span id="feedbackStatus"></span></section><script>
let latest=null; const $=id=>document.getElementById(id); $('image').onchange=e=>{const f=e.target.files[0];$('preview').style.display=f?'block':'none';if(f)$('preview').src=URL.createObjectURL(f)};
function error(t){$('error').hidden=!t;$('error').textContent=t||''} function pct(x){return (x*100).toFixed(1)+'%'}
$('predict').onclick=async()=>{const f=$('image').files[0];if(!f)return error('กรุณาเลือกภาพก่อน');error('');$('predict').disabled=true;$('predict').textContent='กำลังวิเคราะห์…';try{const d=new FormData();d.append('image',f);const r=await fetch('/predict',{method:'POST',body:d});const x=await r.json();if(!r.ok)throw Error(typeof x.detail==='string'?x.detail:JSON.stringify(x.detail));latest=x;$('result').hidden=false;$('label').textContent=x.predicted_class;$('confidence').textContent='confidence '+pct(x.confidence);$('review').innerHTML=x.review_required?'<p class="warning">ควรส่งผลนี้ให้ผู้เชี่ยวชาญตรวจทาน</p>':'';$('scores').innerHTML=Object.entries(x.scores).map(([k,v])=>`<div class="score"><span>${k}</span><div class="bar"><i style="width:${v*100}%"></i></div><b>${pct(v)}</b></div>`).join('');$('meta').textContent=`${x.model_version} · preprocessing ${x.preprocessing_version} · ${x.processing_ms} ms`;$('confirmed').innerHTML=Object.keys(x.scores).map(k=>`<option ${k===x.predicted_class?'selected':''}>${k}</option>`).join('')}catch(e){error(e.message)}finally{$('predict').disabled=false;$('predict').textContent='วิเคราะห์ภาพ'}};
$('sendFeedback').onclick=async()=>{if(!latest)return;const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prediction_id:latest.prediction_id,confirmed_class:$('confirmed').value,note:$('note').value})});$('feedbackStatus').textContent=r.ok?' บันทึกแล้ว':' บันทึกไม่สำเร็จ'};
</script></body></html>'''
