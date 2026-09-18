"""YOLOv5 自动标注配置与执行入口。"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "yolov5"))
import detect

if __name__ == "__main__":
    # 自动标注参数。
    detect.run(
        # 检测权重路径。
        weights=Path(
            "runs/train/exp/weights/best.pt"
        ),  # 使用的检测权重文件。
        # 待标注图片路径。
        source=Path(
            "D:/Study/Datacollect/classified_dataset/all_picturedata/images/train"
        ),  # 待标注图片所在目录。
        # 数据集类别配置路径。
        data=Path(
            "D:/Study/Python/YOLOv5/yolov5/data/myconfigs/auto_aimdevice.yaml"
        ),  # 类别数量和类别名称配置。
        imgsz=(320, 320),  # 推理输入尺寸。
        conf_thres=0.5,  # 置信度阈值。
        iou_thres=0.45,  # NMS 的 IoU 阈值。
        save_txt=True,  # 保存 YOLO 格式标签。
        save_conf=False,  # 不在标签中保存置信度。
        nosave=True,  # 不保存带检测框的图片。
        # 输出目录设置。
        project=Path(
            "runs/auto_label"
        ),  # 自动标注结果目录。
        name="exp",
        exist_ok=True,  # 允许使用已存在的输出目录。
    )

    print("\n 自动标注完成！")
    print("生成的标签文件在 runs/auto_label/exp/labels")
