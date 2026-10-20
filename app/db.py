"""SQLite storage for doctors, sessions, patients and scans (data/app.db + data/images/)."""
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(HERE, "data"))
IMG_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "app.db")
SESSION_DAYS = 7
PBKDF2_ROUNDS = 200_000

os.makedirs(IMG_DIR, exist_ok=True)
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.execute("PRAGMA foreign_keys = ON")
_lock = threading.Lock()

_conn.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
    role TEXT DEFAULT 'Doctor', hospital TEXT DEFAULT '', phone TEXT DEFAULT '',
    pw_salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY, mrn TEXT DEFAULT '', name TEXT NOT NULL, age INTEGER, sex TEXT DEFAULT '',
    phone TEXT DEFAULT '', diabetes_years INTEGER, notes TEXT DEFAULT '',
    created_by INTEGER REFERENCES users(id), created_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id), created_ms INTEGER NOT NULL,
    foot_side TEXT DEFAULT '', pulses TEXT DEFAULT 'unknown',
    ut_class TEXT, grade INTEGER, stage TEXT, grade_conf REAL, p_infection REAL, p_ischemia REAL,
    urgent INTEGER DEFAULT 0, flags TEXT DEFAULT '[]', result TEXT DEFAULT '{}', report TEXT DEFAULT '',
    doctor_notes TEXT DEFAULT '', image_file TEXT DEFAULT '', heatmap_file TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS scans_patient ON scans(patient_id, created_ms);
""")


def now_ms() -> int:
    return int(time.time() * 1000)


def _q(sql, args=(), one=False, commit=False):
    with _lock:
        cur = _conn.execute(sql, args)
        if commit:
            _conn.commit()
            return cur.lastrowid
        rows = cur.fetchall()
    rows = [dict(r) for r in rows]
    return (rows[0] if rows else None) if one else rows


# ---------------- users & sessions ----------------
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()


def create_user(email, password, name, role="Doctor", hospital="", phone=""):
    salt = secrets.token_hex(16)
    return _q("INSERT INTO users(email,name,role,hospital,phone,pw_salt,pw_hash,created_ms) "
              "VALUES(?,?,?,?,?,?,?,?)",
              (email.lower().strip(), name, role, hospital, phone, salt, _hash(password, salt), now_ms()),
              commit=True)


def public_user(u):
    return {k: u[k] for k in ("id", "email", "name", "role", "hospital", "phone")} if u else None


def login(email: str, password: str):
    u = _q("SELECT * FROM users WHERE email=?", (email.lower().strip(),), one=True)
    if not u or not secrets.compare_digest(u["pw_hash"], _hash(password, u["pw_salt"])):
        return None
    token = secrets.token_urlsafe(32)
    _q("INSERT INTO sessions(token,user_id,expires_ms) VALUES(?,?,?)",
       (token, u["id"], now_ms() + SESSION_DAYS * 86_400_000), commit=True)
    return token, public_user(u)


def user_for_token(token: str):
    u = _q("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires_ms>?",
           (token, now_ms()), one=True)
    return public_user(u)


def logout(token: str):
    _q("DELETE FROM sessions WHERE token=?", (token,), commit=True)


def update_user(uid: int, **fields):
    fields = {k: v for k, v in fields.items() if k in ("name", "hospital", "phone") and v is not None}
    if fields:
        _q(f"UPDATE users SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
           (*fields.values(), uid), commit=True)
    return public_user(_q("SELECT * FROM users WHERE id=?", (uid,), one=True))


def seed_default_user():
    """Demo account for development; override with SEED_EMAIL / SEED_PASSWORD."""
    if _q("SELECT COUNT(*) AS n FROM users", one=True)["n"] == 0:
        create_user(os.getenv("SEED_EMAIL", "doctor1@gmail.com"),
                    os.getenv("SEED_PASSWORD", "doctor123"),
                    os.getenv("SEED_NAME", "Doctor 1"))


# ---------------- patients & scans ----------------
PATIENT_FIELDS = ("mrn", "name", "age", "sex", "phone", "diabetes_years", "notes")


def create_patient(data: dict, uid: int):
    cols = [k for k in PATIENT_FIELDS if data.get(k) is not None]
    pid = _q(f"INSERT INTO patients({', '.join(cols)}, created_by, created_ms) "
             f"VALUES({', '.join('?' * len(cols))}, ?, ?)",
             (*[data[k] for k in cols], uid, now_ms()), commit=True)
    return get_patient(pid)


def update_patient(pid: int, data: dict):
    cols = [k for k in PATIENT_FIELDS if k in data]
    if cols:
        _q(f"UPDATE patients SET {', '.join(f'{k}=?' for k in cols)} WHERE id=?",
           (*[data[k] for k in cols], pid), commit=True)
    return get_patient(pid)


_PATIENT_SUMMARY = """
SELECT p.*, COUNT(s.id) AS scan_count, MAX(s.created_ms) AS last_scan_ms,
       (SELECT ut_class FROM scans WHERE patient_id=p.id ORDER BY created_ms DESC LIMIT 1) AS last_class,
       (SELECT urgent FROM scans WHERE patient_id=p.id ORDER BY created_ms DESC LIMIT 1) AS last_urgent
FROM patients p LEFT JOIN scans s ON s.patient_id=p.id
"""


def get_patient(pid: int):
    return _q(_PATIENT_SUMMARY + " WHERE p.id=? GROUP BY p.id", (pid,), one=True)


def list_patients(q: str = ""):
    if q:
        like = f"%{q.strip()}%"
        return _q(_PATIENT_SUMMARY + " WHERE p.name LIKE ? OR p.mrn LIKE ? OR p.phone LIKE ? "
                  "GROUP BY p.id ORDER BY COALESCE(last_scan_ms, p.created_ms) DESC", (like, like, like))
    return _q(_PATIENT_SUMMARY + " GROUP BY p.id ORDER BY COALESCE(last_scan_ms, p.created_ms) DESC")


def delete_patient(pid: int):
    for s in _q("SELECT image_file, heatmap_file FROM scans WHERE patient_id=?", (pid,)):
        for f in (s["image_file"], s["heatmap_file"]):
            if f and os.path.exists(os.path.join(IMG_DIR, f)):
                os.remove(os.path.join(IMG_DIR, f))
    _q("DELETE FROM patients WHERE id=?", (pid,), commit=True)


def _scan_out(s):
    if not s:
        return None
    s = dict(s)
    s["flags"] = json.loads(s["flags"] or "[]")
    s["result"] = json.loads(s["result"] or "{}")
    s["has_image"], s["has_heatmap"] = bool(s.pop("image_file")), bool(s.pop("heatmap_file"))
    s["urgent"] = bool(s["urgent"])
    return s


def save_scan(pid: int, uid: int, out: dict, image: bytes, heatmap_b64: str, foot_side: str, pulses: str):
    import base64
    r = out["result"]
    ts = now_ms()
    img_name = f"scan_{pid}_{ts}.jpg"
    with open(os.path.join(IMG_DIR, img_name), "wb") as f:
        f.write(image)
    hm_name = ""
    if heatmap_b64:
        hm_name = f"scan_{pid}_{ts}_cam.jpg"
        with open(os.path.join(IMG_DIR, hm_name), "wb") as f:
            f.write(base64.b64decode(heatmap_b64))
    urgent = int(r["grade"]["value"] >= 2 or r["ischemia"]["positive"]
                 or "possible_ischemia_check_pulses_abi" in out["flags"])
    sid = _q("INSERT INTO scans(patient_id,user_id,created_ms,foot_side,pulses,ut_class,grade,stage,grade_conf,"
             "p_infection,p_ischemia,urgent,flags,result,report,image_file,heatmap_file) "
             "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             (pid, uid, ts, foot_side, pulses, r["ut_class"], r["grade"]["value"], r["stage"]["value"],
              r["grade"]["confidence"], r["infection"]["prob"], r["ischemia"]["prob"], urgent,
              json.dumps(out["flags"]), json.dumps(r), out.get("report", ""), img_name, hm_name),
             commit=True)
    return get_scan(sid)


def get_scan(sid: int):
    return _scan_out(_q("SELECT * FROM scans WHERE id=?", (sid,), one=True))


def scan_file(sid: int, kind: str):
    s = _q("SELECT image_file, heatmap_file FROM scans WHERE id=?", (sid,), one=True)
    if not s:
        return None
    name = s["heatmap_file"] if kind == "heatmap" else s["image_file"]
    path = os.path.join(IMG_DIR, name) if name else ""
    return path if path and os.path.exists(path) else None


def patient_scans(pid: int):
    return [_scan_out(s) for s in _q("SELECT * FROM scans WHERE patient_id=? ORDER BY created_ms DESC", (pid,))]


def recent_scans(limit: int = 10):
    rows = _q("SELECT s.*, p.name AS patient_name FROM scans s JOIN patients p ON p.id=s.patient_id "
              "ORDER BY s.created_ms DESC LIMIT ?", (limit,))
    return [_scan_out(r) for r in rows]


def set_doctor_notes(sid: int, notes: str):
    _q("UPDATE scans SET doctor_notes=? WHERE id=?", (notes, sid), commit=True)
    return get_scan(sid)


def stats():
    start_of_day = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1)) * 1000
    one = lambda sql, a=(): _q(sql, a, one=True)["n"]  # noqa: E731
    return {
        "patients": one("SELECT COUNT(*) AS n FROM patients"),
        "scans": one("SELECT COUNT(*) AS n FROM scans"),
        "scans_today": one("SELECT COUNT(*) AS n FROM scans WHERE created_ms>=?", (start_of_day,)),
        "urgent_patients": one(
            "SELECT COUNT(*) AS n FROM patients p WHERE (SELECT urgent FROM scans WHERE patient_id=p.id "
            "ORDER BY created_ms DESC LIMIT 1)=1"),
    }
