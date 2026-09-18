# 个人 YOLOv5 视觉开发工作流

这是一个以 YOLOv5 为基础整理的个人视觉开发工作区，主要用于目标检测模型训练、MaixCam 专用 ONNX 导出，以及将 ONNX 交给独立的 Docker 量化流程。

仓库保留了 YOLOv5 的训练和检测代码作为基础，同时增加了项目配置和工作流工具。它不是 Ultralytics 官方 YOLOv5 仓库。

## 当前状态

- 训练和自动标注使用本地 `.trainenv` 环境。
- MaixCam ONNX 导出使用独立的 `.exportenv` 环境。
- 已使用 NVIDIA GeForce RTX 5070 Ti Laptop GPU 完成 GPU 冒烟测试。
- 测试生成了 `runs/train/maixcam_test/weights/best.pt` 和 `runs/export/maixcam_test/best.onnx`。
- 该 ONNX 已在独立 Docker 量化流程中完成测试。
- 当前没有接入 YOLOv8/YOLO11，仓库目前专注于 YOLOv5 和 MaixCam。

## 主要工作流

编辑 [configs/maixcam.yaml](configs/maixcam.yaml)，然后在 VSCode 中打开 [tools/maixcam_pipeline.py](tools/maixcam_pipeline.py)，点击 **Run Code**。脚本会自动调用两个 Windows 虚拟环境：

```text
.trainenv  -> YOLOv5 训练
.exportenv -> MaixCam ONNX 导出
```

导出过程会截断 `Detect` 头，生成 MaixCam 转换流程需要的三个原始特征图输出：

```text
输入：[1, 3, 160, 256]
输出：out0、out1、out2
```

原有入口仍然保留：

```text
start_train.py  -> .trainenv
auto_label.py   -> .trainenv
start_export.py -> .exportenv
```

## 依赖环境

两套环境分别安装依赖：

```powershell
.trainenv\Scripts\python.exe -m pip install -r requirements_train.txt
.exportenv\Scripts\python.exe -m pip install -r requirements_convert.txt
```

虚拟环境只保存在本机，不纳入 Git。

## Docker 交接

Docker 量化脚本独立维护在：

```text
D:\docker_data\models\convert_yolov5_to_cvimodel.sh
```

Python 工作流将正式 ONNX 保存到 `runs/export/`，并将一个可配置名称的副本复制到 `D:\docker_data\models`。它不会启动 Docker、修改量化脚本，也不会管理生成的 `.cvimodel` 文件。

在 `tpu-env` 中根据当前模型手动修改量化参数后执行：

```bash
chmod +x convert_yolov5_to_cvimodel.sh
./convert_yolov5_to_cvimodel.sh
```

详细交接说明见 [MAIXCAM_WORKFLOW.md](MAIXCAM_WORKFLOW.md)，整理过程见 [PLAN_PROGRESS.md](PLAN_PROGRESS.md)。

## 仓库说明

这是一个持续演进的个人工作区。数据集、模型权重、虚拟环境、训练产物、Docker 输出和机器相关路径均不纳入版本控制；公开仓库主要记录可复用的工作流和配置结构。
