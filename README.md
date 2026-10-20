# UT Diabetic Foot Ulcer API (FastAPI)

Serves the trained `.pth` and runs the agent pipeline:
**quality gate → classifier → Grad-CAM explainer → triage rules → report**.

## Run locally (Windows, this PC)

```powershell
cd "D:\Foot ulcer images\app\backend"
pip install -r requirements.txt            # torch/torchvision already installed
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Docs/try-it page: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Phone on the same Wi-Fi: use `http://<this PC's IP>:8000` in the app's Settings
  (find the IP with `ipconfig`, the "IPv4 Address" of your Wi-Fi adapter). Allow Python
  through Windows Firewall when asked.

Smoke test without a server:

```powershell
python -m tests.smoke_test "D:\Foot ulcer images\train_grade_2B\train_grade_2B_01.jpg"
```

## Swap in the real model

`weights/` currently holds `utnet_b0_subset20_fp16.pth` (278-image demo model, not accurate).
After the Kaggle run, copy `utnet_b0_v1_fp16.pth` into `weights/` and restart. The newest
`.pth` is loaded automatically, or set `MODEL_PATH`. Temperature, thresholds and input size
are read from the checkpoint.

## Accounts, patients and storage

- Doctors sign in with `POST /v1/auth/login` (`{email, password}`) and send `Authorization: Bearer <token>`
  on every `/v1/*` call. Sessions last 7 days; passwords are stored as PBKDF2 hashes.
- On first start a demo account is created: **doctor1@gmail.com / doctor123**
  (override with `SEED_EMAIL`, `SEED_PASSWORD`, `SEED_NAME`). Change it before real use.
- Patients, scans, photos and heatmaps are stored in `data/` (`app.db` SQLite + `images/`).
  Set `DATA_DIR` to move it. This folder holds patient data: back it up and keep it private.
- Endpoints: `/v1/me`, `/v1/stats`, `/v1/patients` (list/search, create, get with scans, update, delete),
  `/v1/scans/{id}`, `/v1/scans/{id}/notes`, `/v1/scans/{id}/image?kind=photo|heatmap`.
  `POST /v1/analyze` saves the scan to the patient when `patient_id` is sent.

## API

`POST /v1/analyze` (multipart): `image` (file), `lang` (`en|bn|hi`),
`pulses_palpable` (`yes|no|unknown`), `explain` (bool). Header `Authorization: Bearer <API_KEY>`.

Response:

```json
{ "status": "ok | retake | rejected | error",
  "message": "tip shown to the user when status != ok",
  "quality": {"sharpness": 1637.4, "brightness": 169.0},
  "result": { "ut_class": "1B",
              "grade": {"value": 1, "confidence": 0.45, "probs": [..4..], "meaning": "..."},
              "stage": {"value": "B", "meaning": "infected"},
              "infection": {"prob": 0.62, "positive": true, "threshold": 0.5},
              "ischemia":  {"prob": 0.40, "positive": false, "threshold": 0.5},
              "model_version": "utnet-b0-subset20" },
  "flags": ["possible_infection", "low_confidence_needs_review"],
  "heatmap_jpeg": "<base64>",
  "report": "Texas class 1B: ...",
  "timings_ms": {"quality": 67, "classify": 573, "explain": 870, "triage": 0, "report": 0} }
```

## Optional AI summary

Set `ANTHROPIC_API_KEY` to have Claude (`claude-opus-5-5`, low effort, server-side
refusal fallback enabled) write the summary in the chosen language. Only the structured
result is sent - never the photo or patient details. Without a key, or on any error,
timeout or refusal, a fixed English template is used. See `.env.example`.

## Deploy (hosting)

The `Dockerfile` is ready for any container host (Render, Railway, Fly.io, Google Cloud Run, a VM):

```bash
docker build -t dfu-api .
docker run -p 8000:8000 -v dfu-data:/srv/data \
  -e SEED_EMAIL=you@clinic.org -e SEED_PASSWORD='a-strong-password' dfu-api
```

Checklist before real use:

1. **Set `SEED_EMAIL` / `SEED_PASSWORD` on the host before the first start.** Otherwise the
   public demo login `doctor1@gmail.com / doctor123` is created and anyone can sign in.
   (The seed runs only when the database has no users.)
2. **Attach a persistent disk at `/srv/data`** (`DATA_DIR`). Without it, patients and scans are
   lost on every redeploy/restart. On Render: Disks → mount path `/srv/data`.
3. **RAM:** about 600 MB per worker (PyTorch + model). Keep `WEB_CONCURRENCY=1` on 512 MB-1 GB plans.
4. The host's `$PORT` is used automatically. Serve over **HTTPS** (hosts do this for you),
   then set `usesCleartextTraffic="false"` in the app's AndroidManifest and point the app at
   the https URL (login screen → server icon, or build with `--dart-define=API_URL=...`).
5. Hosting the Flutter **web** build on another domain: set `CORS_ORIGINS=https://your-web-app`.
6. Swap in the Kaggle-trained model: put its `.pth` in `weights/` (newest file is loaded).

Tunables in `app/orchestrator.py`: `MIN_SHARPNESS`, `BRIGHTNESS_RANGE`, `LOW_CONFIDENCE`.
