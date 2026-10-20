"""Agent pipeline: quality gate -> classifier -> explainer -> triage rules -> report.

Critical steps stop the case on failure; optional steps degrade gracefully."""
import asyncio
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.report import write_report

MIN_SHARPNESS = 40.0        # Laplacian variance at 512 px width; calibrate on your photos
BRIGHTNESS_RANGE = (40.0, 220.0)
LOW_CONFIDENCE = 0.55


@dataclass
class Case:
    image: bytes
    lang: str = "en"
    pulses: str = "unknown"                 # yes | no | unknown (answered in the app)
    status: str = "ok"                      # ok | retake | rejected | error
    message: str = ""
    quality: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    heatmap_jpeg: str = ""
    report: str = ""
    timings_ms: dict = field(default_factory=dict)
    x: object = None                        # model input tensor


class Orchestrator:
    def __init__(self, steps):
        self.steps = steps                   # [(name, async fn, critical)]

    async def run(self, image: bytes, lang: str = "en", pulses: str = "unknown",
                  explain: bool = True) -> dict:
        c = Case(image, lang, pulses)
        for name, step, critical in self.steps:
            if name == "explain" and not explain:
                continue
            t0 = time.perf_counter()
            try:
                await step(c)
            except Exception as e:           # noqa: BLE001 - one agent must not crash the pipeline
                if critical:
                    c.status, c.message = "error", f"{name} failed: {type(e).__name__}"
                    c.flags.append(f"{name}_failed")
                else:
                    c.flags.append(f"{name}_skipped")
            c.timings_ms[name] = round((time.perf_counter() - t0) * 1000)
            if c.status != "ok":
                break
        keys = ("status", "message", "quality", "result", "flags", "heatmap_jpeg", "report",
                "timings_ms")
        return {k: getattr(c, k) for k in keys}


def default_steps(clf):
    async def quality(c: Case):
        g = cv2.imdecode(np.frombuffer(c.image, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None:
            c.status, c.message = "rejected", "The file is not a readable image."
            c.flags.append("unreadable_image"); return
        g = cv2.resize(g, (512, max(1, int(512 * g.shape[0] / g.shape[1]))))
        sharp, bright = float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())
        c.quality = {"sharpness": round(sharp, 1), "brightness": round(bright, 1)}
        if sharp < MIN_SHARPNESS:
            c.status, c.message = "retake", "Photo looks blurry. Hold the phone steady and tap to focus."
            c.flags.append("blurry_photo")
        elif not BRIGHTNESS_RANGE[0] < bright < BRIGHTNESS_RANGE[1]:
            c.status = "retake"
            c.message = ("Photo is too dark. Use daylight or room light, avoid shadows."
                         if bright <= BRIGHTNESS_RANGE[0] else
                         "Photo is too bright. Avoid flash glare and direct sunlight.")
            c.flags.append("poor_lighting")

    async def classify(c: Case):
        c.x = clf.tensor(c.image)
        c.result = await asyncio.to_thread(clf.predict, c.x)

    async def explain(c: Case):
        c.heatmap_jpeg = await asyncio.to_thread(clf.gradcam_jpeg, c.x, c.result["grade"]["value"])

    async def triage(c: Case):
        r = c.result
        if r["grade"]["value"] >= 2:
            c.flags.append("deep_ulcer_urgent_referral")
        if r["ischemia"]["positive"] or c.pulses == "no":
            c.flags.append("possible_ischemia_check_pulses_abi")
        if r["infection"]["positive"]:
            c.flags.append("possible_infection")
        if r["grade"]["confidence"] < LOW_CONFIDENCE:
            c.flags.append("low_confidence_needs_review")

    async def report(c: Case):
        c.report = await write_report(c.result, c.flags, c.lang)

    return [("quality", quality, True), ("classify", classify, True),
            ("explain", explain, False), ("triage", triage, True), ("report", report, False)]
