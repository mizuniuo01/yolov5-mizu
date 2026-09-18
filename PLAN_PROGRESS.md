# 本地视觉开发环境整理推进记录

> 整理已完成。该文件保留为本地工作流记录，后续可按需要改名为正式文档。

## 当前状态

已完成本地仓库整理、双环境工作流、GPU 1 epoch 测试、MaixCam ONNX 导出、Docker 输入复制和本地 Git 历史重写。

## 目标

把当前 YOLOv5 仓库整理成可以发布到新远程仓库的本地版本，并增加一个更上位的 MaixCam 工作流入口：

```text
训练集与配置
    -> YOLOv5 训练
    -> MaixCam 专用 Detect 截断导出
    -> 工程 runs/export/ 保存 ONNX
    -> 复制 ONNX 和样本到 D:/docker_data/models
```

Docker Desktop 中的 `tpu-env` 和 `convert_yolov5_to_cvimodel.sh` 保持独立。上位脚本不启动 Docker、不进入容器、不修改量化脚本、不收集或清理 `.cvimodel`。

## 已确认的工作边界

- 只整理当前本地仓库；不操作现有远程仓库。
- 后续由用户自行删除旧远程并创建新远程。
- YOLOv5 原始检测训练代码基本保留。
- 原始 `start_train.py` 和 `start_export.py` 保留为独立兼容入口。
- 上位脚本是额外的省事入口，不取代原脚本。
- 自动标注继续独立，不并入训练加导出上位流程。
- 工程目录保存正式 ONNX；`D:/docker_data/models` 保存给 Docker 使用的副本。
- 输入尺寸约定为宽 256、高 160，ONNX 张量为 `[1, 3, 160, 256]`。
- MaixCam ONNX 输出名为 `out0`、`out1`、`out2`。
- 量化脚本继续由用户手动修改参数并在 Docker 终端执行：

  ```bash
  chmod +x convert_yolov5_to_cvimodel.sh
  ./convert_yolov5_to_cvimodel.sh
  ```

## 已完成

- 新增 `requirements_train.txt`。
- 新增 `requirements_convert.txt`。
- 已确认 `.trainenv` 专用于训练和自动标注，`.exportenv` 专用于 MaixCam ONNX 导出。
- 删除原版 `requirements.txt`。
- 将 YOLOv5 内部的依赖检查逐步改为 `requirements_train.txt`。
- 新增 `configs/maixcam.yaml`。
- 新增 `tools/maixcam_pipeline.py`。
- 新增 `tools/maixcam_export.py`，供上位脚本复用 MaixCam 截断导出逻辑。
- 恢复并保留原始 `start_export.py`。
- 保留原始 `start_train.py`。
- 新增 `MAIXCAM_WORKFLOW.md` 初版说明。
- `.gitignore` 已增加配置、工具目录和本地虚拟环境忽略项。
- 配置 YAML 已通过解析检查；新增 Python 文件已通过 AST 解析检查。

## 待完成工作

### 1. 先清理当前工作区状态

- 处理当前大量 Windows 换行符造成的全仓库伪修改。
- 检查 `.gitattributes`，统一文本文件换行规则。
- 确认 `.trainenv` 和 `.exportenv` 不再被 Git 跟踪。
- 检查当前被删除或修改的个人配置，尤其是 `data/myconfigs/auto_aimdevice.yaml`。
- 不覆盖用户已有的真实代码修改。

### 2. 完善上位脚本

- 训练阶段应能选择正确的训练 Python 环境。
- 转换阶段应能选择正确的转换 Python 环境。
- 不能假设当前 shell 已经激活正确环境。
- 校验数据集 YAML、初始权重和导出权重是否存在。
- 训练完成后可靠定位本次 `best.pt`，不能只拼接固定的 `exp` 名称。
- 默认从数据集 YAML 的 `train` 路径抽取样本。
- 随机抽取固定数量的 `test0.jpg` 到 `test9.jpg`。
- 抽取校准图片到 `D:/docker_data/models/images/`。
- 避免旧的 `test*.jpg` 和校准图片造成误用，必要时明确提示或覆盖策略。
- 复制 Docker 输入失败时给出清晰错误，不影响工程内 ONNX 的保存。
- 上位脚本不得启动或修改 Docker。

### 3. 修正尺寸和原始入口说明

- 原始 `start_export.py` 当前是用户已有的独立脚本，不能被上位脚本替换。
- 需要在文档中明确原始脚本和上位脚本的差异。
- 确认原始脚本的 `torch.randn` 高宽顺序是否也要改为 `[1,3,160,256]`；修改前必须以 MaixCam 实际输入约定为准。
- 上位脚本的导出逻辑必须和原始脚本保持一致：截断 `Detect.forward`，输出三个纯卷积特征图。

### 4. 依赖和文档收尾

- 检查所有 Python、README、Docker 配置对旧 `requirements.txt` 的引用。
- 决定是否保留原 YOLOv5 的分类、分割和 Docker 辅助代码；基本保留源码，但删除明显无关的上游发布配置。
- 更新中文 README 或正式工作流文档，说明新电脑初始化、训练、导出、自动标注和 Docker 交接。
- 补充依赖版本与当前两个环境的对应关系。

### 5. CI 整理

- 删除上游仓库专用的 CLA、自动合并、自动评论、发布镜像等工作流。
- 保留轻量检查：配置解析、Python 导入、路径逻辑和导出适配器检查。
- CI 不执行 GPU 训练，不依赖 Docker Desktop 或 `tpu-env`。

### 6. 最终验证和本地 Git 历史

- 在不训练和不调用 Docker 的前提下完成静态验证。
- 使用一个临时测试配置验证路径解析、输出目录和 Docker 副本逻辑。
- 确认 Git 工作区只剩预期改动。
- 先创建旧历史备份引用。
- 在本地生成干净提交树，不推送远程。
- 用户确认后再自行创建新远程仓库。

## 明确不做

- 不自动启动 Docker Desktop。
- 不自动执行 `docker start tpu-env` 或 `docker attach tpu-env`。
- 不自动修改 `convert_yolov5_to_cvimodel.sh`。
- 不自动收集、移动或删除 `.cvimodel`。
- 不把 YOLOv8/YOLO11 接入当前实现；只保留以后复用整理方法的文档边界。
- 不在没有确认的情况下大规模删除 YOLOv5 源码。
- 不直接重写或强制推送远程 Git 历史。
- 不把 Docker 目录中的模型名写死为 `steel_ball`；名称来自配置或导出权重文件名。
- 上位入口以 `tools/maixcam_pipeline.py` 的 VSCode `Run Code` 为主，不要求用户执行 PowerShell 启动器。

## 当前工作量估计

- 上位脚本完善与静态测试：中等工作量。
- 仓库文件和换行清理：中等工作量，需谨慎保护已有改动。
- CI 与文档整理：小到中等工作量。
- 本地 Git 历史重写：最后单独执行，属于高风险步骤，必须在工作区最终确认后进行。

## 完成记录

- GPU 测试使用 `NVIDIA GeForce RTX 5070 Ti Laptop GPU`，训练阶段确认 `CUDA:0`，显存约 8.8 GB。
- 测试权重：`runs/train/maixcam_test/weights/best.pt`。
- 测试 ONNX：`runs/export/maixcam_test/best.onnx`。
- Docker 副本：`D:/docker_data/models/best.onnx`。
- Docker 校准输入：10 张 `test*.jpg` 和 197 张 `images/` 图片。
- Docker 量化测试：已使用上述 ONNX 在 `tpu-env` 中完成测试，结果成功。
- 本地新根提交：`48ee934`。
- 旧整理提交备份：`backup-before-history-rewrite`。
- 原始历史备份：`before-local-cleanup`。
