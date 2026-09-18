# Personal YOLOv5 Vision Workflow

This repository is a personal computer vision development workspace based on YOLOv5. It is organized around training object detection models, exporting the MaixCam-specific ONNX representation, and handing that ONNX file to a separate Docker quantization workflow.

The repository keeps the original YOLOv5 training and detection code as a base, while adding project configuration and workflow tools. It is not the upstream Ultralytics YOLOv5 repository.

## Current status

- Training and auto-labeling use the local `.trainenv` environment.
- MaixCam ONNX export uses the separate `.exportenv` environment.
- A GPU smoke test completed with an NVIDIA GeForce RTX 5070 Ti Laptop GPU.
- The test produced `runs/train/maixcam_test/weights/best.pt` and `runs/export/maixcam_test/best.onnx`.
- The ONNX file was successfully used in the separate Docker quantization test.
- YOLOv8 and YOLO11 are not integrated here yet; the current workflow is intentionally focused on YOLOv5 and MaixCam.

## Main workflow

Edit [configs/maixcam.yaml](configs/maixcam.yaml), then open [tools/maixcam_pipeline.py](tools/maixcam_pipeline.py) in VSCode and use **Run Code**. The script calls the two Windows virtual environments itself:

```text
.trainenv  -> train YOLOv5
.exportenv -> export MaixCam ONNX
```

The export uses a truncated `Detect` head and produces the three raw feature-map outputs required by the MaixCam conversion flow:

```text
input:  [1, 3, 160, 256]
outputs: out0, out1, out2
```

The original entry points remain available:

```text
start_train.py  -> .trainenv
auto_label.py   -> .trainenv
start_export.py -> .exportenv
```

## Dependencies

Install the environments separately:

```powershell
.trainenv\Scripts\python.exe -m pip install -r requirements_train.txt
.exportenv\Scripts\python.exe -m pip install -r requirements_convert.txt
```

The virtual environments themselves are local files and are intentionally not tracked by Git.

## Docker handoff

The Docker quantization script is maintained outside this repository at:

```text
D:\docker_data\models\convert_yolov5_to_cvimodel.sh
```

The Python workflow saves the canonical ONNX under `runs/export/` and copies a configurable-name copy to `D:\docker_data\models`. It does not start Docker, edit the shell script, or manage the resulting `.cvimodel` files.

Inside `tpu-env`, after manually updating the quantization parameters for the current model, run:

```bash
chmod +x convert_yolov5_to_cvimodel.sh
./convert_yolov5_to_cvimodel.sh
```

See [MAIXCAM_WORKFLOW.md](MAIXCAM_WORKFLOW.md) for the detailed handoff and [PLAN_PROGRESS.md](PLAN_PROGRESS.md) for the project record.

## Repository policy

This is a personal, evolving workspace. Datasets, model weights, virtual environments, generated runs, Docker outputs, and machine-specific paths are excluded from version control. The public repository records the reproducible workflow and configuration structure, while the actual datasets and deployment tools remain local.
