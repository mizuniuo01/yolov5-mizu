"""导出 MaixCam 所需的 YOLOv5 原始特征图。"""

from __future__ import annotations
import types
from pathlib import Path
import torch
from models.experimental import attempt_load
from models.yolo import Detect


def custom_forward(self, x):
    """返回 Detect 模块中三个卷积层的输出。"""
    return tuple(self.m[i](x[i]) for i in range(self.nl))


def export_pure(
    weights_path,
    onnx_path=None,
    width=256,
    height=160,
    opset=12,
    output_names=("out0", "out1", "out2"),
):
    """将 YOLOv5 权重导出为 MaixCam 使用的 ONNX 文件。

    参数:
        weights_path: 输入的 YOLOv5 权重文件。
        onnx_path: 输出 ONNX 路径；为空时使用权重文件旁的默认名称。
        width: 输入图像宽度。
        height: 输入图像高度。
        opset: ONNX 算子集版本。
        output_names: 原始特征图输出名称。

    返回:
        生成的 ONNX 文件路径。
    """
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
