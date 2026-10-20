"""End-to-end check of the API without starting a server (uses a throwaway database).
Usage (from app/backend):  python -m tests.smoke_test "<path to image>" [...]"""
import os
import sys
import tempfile
import time

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dfu_test_")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

c = TestClient(app)
print("health:", c.get("/health").json())
assert c.post("/v1/auth/login", json={"email": "doctor1@gmail.com", "password": "wrong"}).status_code == 401
assert c.get("/v1/stats").status_code == 401
r = c.post("/v1/auth/login", json={"email": "Doctor1@gmail.com ", "password": "doctor123"})
assert r.status_code == 200, r.text
H = {"Authorization": f"Bearer {r.json()['token']}"}
print("login ok:", r.json()["user"])

p = c.post("/v1/patients", headers=H, json={"name": "Test Patient", "age": 61, "sex": "M", "mrn": "T-001"}).json()
print("patient:", p["id"], p["name"])

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        t0 = time.perf_counter()
        r = c.post("/v1/analyze", headers=H, files={"image": (os.path.basename(path), f, "image/jpeg")},
                   data={"patient_id": str(p["id"]), "foot_side": "left", "pulses_palpable": "unknown"})
    out = r.json()
    print(f"\n{os.path.basename(path)}  HTTP {r.status_code}  {time.perf_counter() - t0:.2f}s  "
          f"status={out['status']} class={out['result'].get('ut_class')} scan={out['scan'] and out['scan']['id']}")
    print("  flags:", out["flags"], "| report:", out["report"][:90], "...")

detail = c.get(f"/v1/patients/{p['id']}", headers=H).json()
print("\npatient scans:", [(s["id"], s["ut_class"], s["urgent"]) for s in detail["scans"]])
if detail["scans"]:
    sid = detail["scans"][0]["id"]
    assert c.get(f"/v1/scans/{sid}/image", headers=H).status_code == 200
    assert c.get(f"/v1/scans/{sid}/image?kind=heatmap", headers=H).status_code == 200
    assert c.put(f"/v1/scans/{sid}/notes", headers=H, json={"doctor_notes": "Debrided."}).json()["doctor_notes"] == "Debrided."
print("stats:", {k: v for k, v in c.get("/v1/stats", headers=H).json().items() if k != "recent_scans"})
print("ALL CHECKS PASSED")
