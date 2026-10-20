"""Loads the .pth once and serves predictions + Grad-CAM heatmaps."""
import base64
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T

from app.net import UTNet

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


class _GradeOnly(nn.Module):                       # Grad-CAM needs a single output tensor
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x)[0]


class Classifier:
    def __init__(self, path: str, threads: int = 2):
        torch.set_num_threads(threads)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.model = UTNet(ck["arch"], pretrained=False)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                    for k, v in ck["state_dict"].items()})
        self.model.eval()
        self.mean, self.std = np.array(ck["mean"]), np.array(ck["std"])
        self.tf = T.Compose([T.Resize((ck["H"], ck["W"])), T.ToTensor(),
                             T.Normalize(ck["mean"], ck["std"])])
        self.temp = float(ck.get("temperature", 1.0))
        self.thr = ck.get("thresholds", {"infection": 0.5, "ischemia": 0.5})
        self.version = ck.get("version", "utnet")
        self.arch = ck["arch"]

    def warmup(self):
        x = self.tf(Image.new("RGB", (256, 576), (128, 90, 80))).unsqueeze(0)
        self.gradcam_jpeg(x, self.predict(x)["grade"]["value"])

    def tensor(self, data: bytes) -> torch.Tensor:
        return self.tf(to_portrait(Image.open(io.BytesIO(data)))).unsqueeze(0)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> dict:
        g1, s1 = self.model(x)
        g2, s2 = self.model(torch.flip(x, dims=[3]))                    # flip TTA
        pg = (((g1 / self.temp).softmax(1) + (g2 / self.temp).softmax(1)) / 2)[0]
        ps = ((s1.sigmoid() + s2.sigmoid()) / 2)[0]
        grade, inf, isc = int(pg.argmax()), float(ps[0]), float(ps[1])
        pi, pc = inf >= self.thr["infection"], isc >= self.thr["ischemia"]
        stage = "D" if pi and pc else "B" if pi else "C" if pc else "A"
        return {
            "ut_class": f"{grade}{stage}",
            "grade": {"value": grade, "confidence": round(float(pg[grade]), 3),
                      "probs": [round(float(p), 3) for p in pg], "meaning": GRADE_TEXT[grade]},
            "stage": {"value": stage, "meaning": STAGE_TEXT[stage]},
            "infection": {"prob": round(inf, 3), "positive": pi,
                          "threshold": round(float(self.thr["infection"]), 3)},
            "ischemia": {"prob": round(isc, 3), "positive": pc,
                         "threshold": round(float(self.thr["ischemia"]), 3)},
            "model_version": self.version,
        }

    def gradcam_jpeg(self, x: torch.Tensor, grade: int) -> str:
        """Base64 JPEG of the input with a Grad-CAM overlay for the predicted grade."""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        with GradCAM(model=_GradeOnly(self.model),
                     target_layers=[self.model.backbone.blocks[-1]]) as cam:
            heat = cam(input_tensor=x.clone(), targets=[ClassifierOutputTarget(grade)])[0]
        rgb = (x[0].permute(1, 2, 0).numpy() * self.std + self.mean).clip(0, 1).astype(np.float32)
        vis = show_cam_on_image(rgb, heat, use_rgb=True)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode() if ok else ""
