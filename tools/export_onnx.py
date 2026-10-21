"""Convert a trained UTNet .pth into the .onnx file the server runs (no PyTorch needed at runtime).

Usage (from app/backend, needs requirements-export.txt):
    python -m tools.export_onnx weights/utnet_b0_v1_fp16.pth
writes weights/utnet_b0_v1.onnx next to it and checks it against PyTorch.

Outputs of the ONNX graph:
    grade_logits (N,4), stage_logits (N,2)   - same as UTNet.forward
    grade_cams   (N,4,h,w)                   - class activation maps for each grade:
        the grade head's weights applied to the last feature map. Because UTNet is
        features -> global average pool -> linear, this is exact CAM (no gradients needed).
Checkpoint metadata (temperature, thresholds, input size, normalisation) is stored in the
ONNX metadata so one file carries everything the server needs.
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from tools.net import UTNet


class _Export(nn.Module):
    def __init__(self, m: UTNet):
        super().__init__()
        self.m = m

    def forward(self, x):
        f = self.m.backbone.forward_features(x)                    # N,C,h,w (after conv_head+bn+act)
        pooled = self.m.backbone.forward_head(f, pre_logits=True)  # N,C (global average pool)
        w = self.m.grade_head.weight                               # 4,C
        cams = torch.einsum("kc,nchw->nkhw", w, f)                 # N,4,h,w
        return self.m.grade_head(pooled), self.m.stage_head(pooled), cams


def export(pth: str) -> str:
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    model = UTNet(ck["arch"], pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    model.eval()
    wrapper = _Export(model).eval()

    out = os.path.splitext(pth)[0].replace("_fp16", "") + ".onnx"
    x = torch.randn(2, 3, ck["H"], ck["W"])
    torch.onnx.export(
        wrapper, (x,), out, dynamo=False, opset_version=17,
        input_names=["image"], output_names=["grade_logits", "stage_logits", "grade_cams"],
        dynamic_axes={"image": {0: "n"}, "grade_logits": {0: "n"}, "stage_logits": {0: "n"},
                      "grade_cams": {0: "n"}},
    )

    meta = {"version": ck.get("version", "utnet"), "arch": ck["arch"], "H": ck["H"], "W": ck["W"],
            "mean": list(ck["mean"]), "std": list(ck["std"]),
            "temperature": float(ck.get("temperature", 1.0)),
            "thresholds": {k: float(v) for k, v in ck.get("thresholds", {"infection": 0.5, "ischemia": 0.5}).items()}}
    m = onnx.load(out)
    m.metadata_props.add(key="utnet", value=json.dumps(meta))
    onnx.save(m, out)

    # parity check against PyTorch
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    g, s, c = sess.run(None, {"image": x.numpy()})
    with torch.no_grad():
        tg, ts, tc = wrapper(x)
    err = max(np.abs(g - tg.numpy()).max(), np.abs(s - ts.numpy()).max())
    pooled_check = np.abs(c.mean(axis=(2, 3)) + model.grade_head.bias.detach().numpy() - g).max()
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"max |onnx - torch| logits: {err:.2e}   CAM mean+bias == logits: {pooled_check:.2e}")
    assert err < 1e-3 and pooled_check < 1e-3, "ONNX export does not match PyTorch"
    return out


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "weights/utnet_b0_subset20_fp16.pth")
