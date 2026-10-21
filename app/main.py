"""FastAPI entry point: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import glob
import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from app.model import Classifier
from app.orchestrator import Orchestrator, default_steps
from app.report import LLM_ENABLED, REPORT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_UPLOAD = 12 * 1024 * 1024


def _model_path() -> str:
    p = os.getenv("MODEL_PATH")
    if p:
        return p
    found = sorted(glob.glob(os.path.join(HERE, "weights", "*.onnx")), key=os.path.getmtime)
    if not found:
        raise RuntimeError("no .onnx in backend/weights - run: python -m tools.export_onnx <model.pth>")
    return found[-1]                                   # newest model wins


MODEL_PATH = _model_path()
clf = Classifier(MODEL_PATH, threads=int(os.getenv("MODEL_THREADS", os.getenv("TORCH_THREADS", "2"))))
clf.warmup()                                           # so the first real request isn't slow
orch = Orchestrator(default_steps(clf))
db.seed_default_user()
log.info("model %s (%s) loaded from %s | LLM report: %s", clf.version, clf.arch,
         os.path.basename(MODEL_PATH), REPORT_MODEL if LLM_ENABLED else "off (template)")

app = FastAPI(title="UT Diabetic Foot Ulcer API", version="2.0.0")
# Browsers (the Flutter web build) need CORS; the Android app does not.
# Default allows only pages served from this PC; set CORS_ORIGINS for a hosted web app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "").split(",") if o],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------- auth ----------------
def _token(authorization: str) -> str:
    return authorization[7:] if authorization.startswith("Bearer ") else ""


def current_user(authorization: str = Header(default="")) -> dict:
    user = db.user_for_token(_token(authorization))
    if not user:
        raise HTTPException(401, "not signed in")
    return user


class LoginIn(BaseModel):
    email: str
    password: str


class ProfileIn(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    hospital: Optional[str] = Field(None, max_length=200)
    phone: Optional[str] = Field(None, max_length=40)


class PatientIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    mrn: str = Field("", max_length=60)
    age: Optional[int] = Field(None, ge=0, le=130)
    sex: str = Field("", max_length=10)
    phone: str = Field("", max_length=40)
    diabetes_years: Optional[int] = Field(None, ge=0, le=100)
    notes: str = Field("", max_length=2000)


class NotesIn(BaseModel):
    doctor_notes: str = Field("", max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": clf.version, "arch": clf.arch,
            "llm_report": REPORT_MODEL if LLM_ENABLED else None}


@app.post("/v1/auth/login")
def login(body: LoginIn):
    res = db.login(body.email, body.password)
    if not res:
        raise HTTPException(401, "Incorrect email or password")
    token, user = res
    return {"token": token, "user": user}


@app.post("/v1/auth/logout")
def logout(authorization: str = Header(default="")):
    db.logout(_token(authorization))
    return {"ok": True}


@app.get("/v1/me")
def me(user=Depends(current_user)):
    return user


@app.put("/v1/me")
def update_me(body: ProfileIn, user=Depends(current_user)):
    return db.update_user(user["id"], **body.model_dump())


@app.get("/v1/stats")
def stats(user=Depends(current_user)):
    return {**db.stats(), "recent_scans": db.recent_scans(6)}


# ---------------- patients ----------------
@app.get("/v1/patients")
def patients(q: str = "", user=Depends(current_user)):
    return db.list_patients(q)


@app.post("/v1/patients")
def create_patient(body: PatientIn, user=Depends(current_user)):
    return db.create_patient(body.model_dump(), user["id"])


@app.get("/v1/patients/{pid}")
def patient(pid: int, user=Depends(current_user)):
    p = db.get_patient(pid)
    if not p:
        raise HTTPException(404, "patient not found")
    return {**p, "scans": db.patient_scans(pid)}


@app.put("/v1/patients/{pid}")
def update_patient(pid: int, body: PatientIn, user=Depends(current_user)):
    if not db.get_patient(pid):
        raise HTTPException(404, "patient not found")
    return db.update_patient(pid, body.model_dump())


@app.delete("/v1/patients/{pid}")
def delete_patient(pid: int, user=Depends(current_user)):
    db.delete_patient(pid)
    return {"ok": True}


# ---------------- scans ----------------
@app.post("/v1/analyze")
async def analyze(image: UploadFile = File(...),
                  patient_id: Optional[int] = Form(None),
                  foot_side: str = Form(""),
                  lang: str = Form("en"),
                  pulses_palpable: str = Form("unknown"),
                  explain: str = Form("true"),   # str: newer FastAPI turns a missing bool form field into False
                  user=Depends(current_user)):
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image larger than 12 MB")
    if pulses_palpable not in ("yes", "no", "unknown"):
        raise HTTPException(422, "pulses_palpable must be yes, no or unknown")
    if patient_id is not None and not db.get_patient(patient_id):
        raise HTTPException(404, "patient not found")
    out = await orch.run(data, lang=lang, pulses=pulses_palpable,
                         explain=explain.strip().lower() not in ("false", "0", "no", "off"))
    out["scan"] = None
    if patient_id is not None and out["status"] == "ok":
        out["scan"] = db.save_scan(patient_id, user["id"], out, data, out["heatmap_jpeg"],
                                   foot_side, pulses_palpable)
    log.info("analyze status=%s class=%s flags=%s ms=%s scan=%s", out["status"],
             out["result"].get("ut_class"), out["flags"], out["timings_ms"],
             out["scan"]["id"] if out["scan"] else None)
    return out


@app.get("/v1/scans/{sid}")
def scan(sid: int, user=Depends(current_user)):
    s = db.get_scan(sid)
    if not s:
        raise HTTPException(404, "scan not found")
    return {**s, "patient": db.get_patient(s["patient_id"])}


@app.put("/v1/scans/{sid}/notes")
def scan_notes(sid: int, body: NotesIn, user=Depends(current_user)):
    if not db.get_scan(sid):
        raise HTTPException(404, "scan not found")
    return db.set_doctor_notes(sid, body.doctor_notes)


@app.get("/v1/scans/{sid}/image")
def scan_image(sid: int, kind: str = "photo", user=Depends(current_user)):
    path = db.scan_file(sid, "heatmap" if kind == "heatmap" else "photo")
    if not path:
        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/jpeg")
