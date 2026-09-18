"""MaixCam 原始导出入口，请使用 .exportenv 运行。"""

import sys
import os
import torch
import types

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from models.experimental import attempt_load
from models.yolo import Detect


# 拦截 YOLOv5 的后处理
def custom_forward(self, x):
    """跳过 Detect 后处理，只返回原始卷积特征图。"""
    res = []
    for i in range(self.nl):
        # 只输出最底层的纯卷积特征图
        res.append(self.m[i](x[i]))
    return tuple(res)


def export_pure():
    """加载指定权重并导出三个原始特征图。"""
    weights_path = os.path.join(
        current_dir, "runs", "train", "exp2", "weights", "steel_ball.pt"
    )
    print(f"加载模型: {weights_path}")
    model = attempt_load(weights_path, device="cpu")  # 加载模型到CPU
    # 评估模式
    model.eval()

    for m in model.modules():
        if isinstance(m, Detect):
            m.forward = types.MethodType(custom_forward, m)
            print("成功截断 Detect 头")

    dummy_input = torch.randn(1, 3, 256, 160)  # 输入尺寸保持和训练时一致
    onnx_path = weights_path.replace(".pt", "_export.onnx")

    print("正在导出ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=12,
        input_names=["images"],
        # 固定命名输出节点，方便后续使用
        output_names=["out0", "out1", "out2"],
    )
    print(f"导出完成，模型保存在: {onnx_path}")


if __name__ == "__main__":
    export_pure()
