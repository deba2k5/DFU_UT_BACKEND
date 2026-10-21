# UT Diabetic Foot Ulcer API (FastAPI + ONNX Runtime)

Classifies diabetic foot ulcer photos with the University of Texas system and runs the agent pipeline:
**quality gate → classifier → CAM heatmap → triage rules → report**.

The server runs the model with **onnxruntime**, so it needs no PyTorch: dependencies install to
about 190 MB, small enough for Vercel's 500 MB function limit and fast to cold-start.

## Run locally

```powershell
cd app\backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Docs/try-it page: http://localhost:8000/docs · health: http://localhost:8000/health
- Phone on the same Wi-Fi: use `http://<this PC's IP>:8000` in the app (login screen → server icon).
- Smoke test (throwaway database): `pip install httpx` then
  `python -m tests.smoke_test path\to\photo.jpg`

## Model: .pth → .onnx

The server loads the newest `weights/*.onnx`. After training a new model on Kaggle:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-export.txt
python -m tools.export_onnx weights\utnet_b0_v1_fp16.pth     # writes weights\utnet_b0_v1.onnx
```

The export embeds temperature, thresholds, input size and normalisation in the ONNX metadata,
and checks the ONNX output against PyTorch. The graph also outputs per-grade class activation
maps (the grade head's weights applied to the last feature map), so the heatmap needs no
gradients. For this features → average-pool → linear model, that is exact CAM.

`weights/utnet_b0_subset20.onnx` is the 278-image demo model: it works, but it is not accurate.

## Accounts, patients and storage

- Doctors sign in with `POST /v1/auth/login` (`{email, password}`) and send
  `Authorization: Bearer <token>` on every `/v1/*` call. Tokens are HMAC-signed with
  `SECRET_KEY` and last 7 days. Passwords are stored as PBKDF2 hashes.
- On first start a demo account is created: **doctor1@gmail.com / doctor123**.
  Override with `SEED_EMAIL`, `SEED_PASSWORD`, `SEED_NAME`.
- Patients, scans, photos and heatmaps go to `DATA_DIR` (default `data/`: SQLite `app.db` + `images/`).
  This is patient data: keep it private and backed up. It is excluded from git.
- Endpoints: `/v1/me`, `/v1/stats`, `/v1/patients` (list/search, create, get with scans, update,
  delete), `/v1/scans/{id}`, `/v1/scans/{id}/notes`, `/v1/scans/{id}/image?kind=photo|heatmap`,
  `POST /v1/analyze` (multipart `image`, `patient_id`, `foot_side`, `pulses_palpable`, `lang`).

## Deploy

### Vercel (serverless)

`vercel.json` and `api/index.py` are included. Import the repo in Vercel, keep the root as this
folder, and set these environment variables:

| Variable | Why |
|---|---|
| `SECRET_KEY` | **Required.** Long random string. Signs login tokens so they work on every instance. |
| `SEED_EMAIL`, `SEED_PASSWORD` | **Required.** Your login. Otherwise the public demo login is created. |
| `ANTHROPIC_API_KEY` | Optional AI-written summary. |

**Important limitation:** Vercel functions have no permanent disk. The SQLite database and photos
live in `/tmp`, which is wiped whenever an instance stops, and each instance has its own copy.
Classification, login and the heatmap work, but **patient records and scan history will not
persist**. For a real clinic, either host on a server with a disk (below) or move storage to a
hosted database (e.g. Postgres on Neon/Supabase) and object storage (Vercel Blob/S3).
Vercel also limits request bodies to 4.5 MB; the app already uploads photos of about 0.5 MB.

### Docker host with a persistent disk (Render, Railway, Fly.io, a VM)

```bash
docker build -t dfu-api .
docker run -p 8000:8000 -v dfu-data:/srv/data \
  -e SECRET_KEY=... -e SEED_EMAIL=you@clinic.org -e SEED_PASSWORD='a-strong-password' dfu-api
```

The image has no PyTorch and needs about 300 MB of RAM. Mount a persistent disk at `/srv/data`.
The host's `$PORT` is used automatically.

For both: serve over HTTPS, set `usesCleartextTraffic="false"` in the app's AndroidManifest, and
point the app at the https URL. A Flutter web build on another domain also needs
`CORS_ORIGINS=https://your-web-app`.

Tunables in `app/orchestrator.py`: `MIN_SHARPNESS`, `BRIGHTNESS_RANGE`, `LOW_CONFIDENCE`.
