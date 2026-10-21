"""Runs the exported UTNet .onnx with onnxruntime (no PyTorch) and draws CAM heatmaps."""
import base64
import io
import json

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

GRADE_TEXT = {
    0: "pre- or post-ulcerative lesion, completely epithelialised",
    1: "superficial wound, not involving tendon, capsule or bone",
    2: "wound penetrating to tendon or capsule",
    3: "wound penetrating to bone or joint",
}
STAGE_TEXT = {"A": "clean (no infection or ischemia)", "B": "infected", "C": "ischemic",
              "D": "infected and ischemic"}


def to_portrait(img: Image.Image) -> Image.Image:
    """Identical to prep_dataset.py: apply EXIF rotation, then make the photo portrait."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img.rotate(90, expand=True) if img.width > img.height else img


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _jet(v: np.ndarray) -> np.ndarray:
    """Jet colour map for values in [0,1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class Classifier:
    def __init__(self, path: str, threads: int = 2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        meta = json.loads(self.sess.get_modelmeta().custom_metadata_map["utnet"])
        self.H, self.W = int(meta["H"]), int(meta["W"])
        self.mean = np.array(meta["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(meta["std"], np.float32).reshape(3, 1, 1)
        self.temp = float(meta.get("temperature", 1.0))
        self.thr = meta.get("thresholds", {"infection": 0.5, "ischemia": 0.5})
        self.version = meta.get("version", "utnet")
        self.arch = meta.get("arch", "")

    def warmup(self):
        x = self._prep(Image.new("RGB", (self.W, self.H), (128, 90, 80)))
        result, cams = self.predict(x)
        self.cam_jpeg(x, cams, result["grade"]["value"])

    def _prep(self, img: Image.Image) -> np.ndarray:
        # Same as torchvision Resize((H, W)) + ToTensor + Normalize on a PIL image.
        a = np.asarray(img.resize((self.W, self.H), Image.BILINEAR), np.float32) / 255.0
        return ((a.transpose(2, 0, 1) - self.mean) / self.std)[None].astype(np.float32)

    def tensor(self, data: bytes) -> np.ndarray:
        return self._prep(to_portrait(Image.open(io.BytesIO(data))))

    def predict(self, x: np.ndarray):
        """Returns (result dict, CAM maps 4 x h x w averaged over the flip TTA pair)."""
        batch = np.concatenate([x, x[..., ::-1]], axis=0)           # original + horizontal flip
        g, s, cams = self.sess.run(None, {"image": np.ascontiguousarray(batch)})
        pg = _softmax(g / self.temp).mean(axis=0)
        ps = _sigmoid(s).mean(axis=0)
        cam = (cams[0] + cams[1][..., ::-1]) / 2
        grade, inf, isc = int(pg.argmax()), float(ps[0]), float(ps[1])
        pi, pc = inf >= self.thr["infection"], isc >= self.thr["ischemia"]
        stage = "D" if pi and pc else "B" if pi else "C" if pc else "A"
        result = {
            "ut_class": f"{grade}{stage}",
            "grade": {"value": grade, "confidence": round(float(pg[grade]), 3),
                      "probs": [round(float(p), 3) for p in pg], "meaning": GRADE_TEXT[grade]},
            "stage": {"value": stage, "meaning": STAGE_TEXT[stage]},
            "infection": {"prob": round(inf, 3), "positive": bool(pi),
                          "threshold": round(float(self.thr["infection"]), 3)},
            "ischemia": {"prob": round(isc, 3), "positive": bool(pc),
                         "threshold": round(float(self.thr["ischemia"]), 3)},
            "model_version": self.version,
        }
        return result, cam

    def cam_jpeg(self, x: np.ndarray, cams: np.ndarray, grade: int) -> str:
        """Base64 JPEG of the model input with the class activation map for `grade` overlaid."""
        heat = np.maximum(cams[grade], 0)
        heat = heat / heat.max() if heat.max() > 0 else heat
        heat = np.asarray(Image.fromarray(heat.astype(np.float32))           # float32 -> mode "F"
                          .resize((self.W, self.H), Image.BILINEAR)).clip(0, 1)
        rgb = ((x[0] * self.std + self.mean).transpose(1, 2, 0).clip(0, 1) * 255).astype(np.float32)
        vis = (0.55 * rgb + 0.45 * _jet(heat)).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(vis).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
