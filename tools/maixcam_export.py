"""MaixCam-specific YOLOv5 export: expose Detect head feature maps."""

from __future__ import annotations
import types
from pathlib import Path
import torch
from models.experimental import attempt_load
from models.yolo import Detect


def custom_forward(self, x):
    return tuple(self.m[i](x[i]) for i in range(self.nl))


def export_pure(
    weights_path,
    onnx_path=None,
    width=256,
    height=160,
    opset=12,
    output_names=("out0", "out1", "out2"),
):
    weights_path = Path(weights_path).resolve()
    onnx_path = (
        Path(onnx_path)
        if onnx_path
        else weights_path.with_name(f"{weights_path.stem}_export.onnx")
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = attempt_load(str(weights_path), device="cpu")
    model.eval()
    for module in model.modules():
        if isinstance(module, Detect):
            module.forward = types.MethodType(custom_forward, module)
    torch.onnx.export(
        model,
        torch.randn(1, 3, int(height), int(width)),
        str(onnx_path),
        opset_version=int(opset),
        input_names=["images"],
        output_names=list(output_names),
    )
    return onnx_path
