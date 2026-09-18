# MaixCam 工作流

编辑 `configs/maixcam.yaml`，在 VSCode 终端运行：

在 VSCode 中打开 `tools/maixcam_pipeline.py`，直接点击 `Run Code`。脚本会分别调用 `.trainenv\Scripts\python.exe` 和 `.exportenv\Scripts\python.exe`，不需要手动切换解释器。训练和自动标注使用 `.trainenv`，MaixCam 导出使用 `.exportenv`。

脚本负责训练 YOLOv5、导出截断 Detect 头后的 MaixCam ONNX，并把正式文件保存到 `runs/export/`。输入统一为宽 256、高 160，输出名为 `out0`、`out1`、`out2`。使用 `--export-only` 可跳过训练。

如果配置了 `docker.directory`，脚本会把 ONNX 复制到该目录。文件名优先使用 `docker.model_name`，未填写时使用导出文件名的 stem；不会写死 `steel_ball`。配置了图片来源后，脚本会准备 `test0.jpg` 到 `test9.jpg` 及 `images/` 校准图片。它不会启动 Docker、修改量化脚本或收集 `.cvimodel`。

量化脚本继续独立维护在 `D:\docker_data\models\convert_yolov5_to_cvimodel.sh`。根据当前模型手动修改参数后，在 `tpu-env` 的挂载目录执行：

```bash
chmod +x convert_yolov5_to_cvimodel.sh
./convert_yolov5_to_cvimodel.sh
```

上位脚本默认执行训练和导出。只导出已有权重时，可以在 VSCode 的运行配置中把参数设为 `--phase export --weights runs/train/maixcam_test/weights/best.pt`，并使用 `.exportenv` 运行；完整流程仍建议直接点击 `Run Code`。
