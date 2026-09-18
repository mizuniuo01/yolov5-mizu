"""通过 VSCode 的 Run Code 执行 MaixCam 训练和导出流程。"""

from __future__ import annotations

import argparse
import codecs
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_PROGRESS = re.compile(
    rf"^\s*(\d+/\d+)\s+(\S+)\s+({NUMBER})\s+({NUMBER})\s+"
    rf"({NUMBER})\s+(\d+)\s+(\d+):"
)
METRIC_LINE = re.compile(
    rf"^\s*(\S+)\s+(\d+)\s+(\d+)\s+({NUMBER})\s+({NUMBER})\s+"
    rf"({NUMBER})\s+({NUMBER})\s*$"
)
MODEL_ROW = re.compile(r"^\s*\d+\s+-?\d+\s+\d+\s+\S+")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path):
    """读取 YAML 工作流配置，并记录配置文件的绝对路径。"""
    import yaml

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_path"] = path.resolve()
    return config


def resolve(value, base=ROOT):
    """将配置中的相对路径按照指定根目录解析为绝对路径。"""
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def image_files(path):
    """按稳定顺序返回目录下支持的图片文件。"""
    return sorted(
        p
        for p in path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def format_output_line(line):
    """将训练输出转换为紧凑的状态行。"""
    line = ANSI_ESCAPE.sub("", line).strip()
    if not line:
        return ""
    match = TRAIN_PROGRESS.match(line)
    if match:
        epoch, gpu, box, obj, cls, instances, size = match.groups()
        return (
            f"Epoch {epoch} | GPU {gpu} | box {box} | obj {obj} | "
            f"cls {cls} | instances {instances} | size {size}"
        )

    match = METRIC_LINE.match(line)
    if match:
        name, images, instances, precision, recall, map50, map95 = match.groups()
        return (
            f"Val {name} | images {images} | instances {instances} | "
            f"P {precision} | R {recall} | mAP50 {map50} | mAP50-95 {map95}"
        )

    if "%|" in line and "Scanning" in line:
        return line.split("%|", 1)[0].strip() + "% 完成"
    if "%|" in line:
        return None
    if line.startswith("TRAIN_WEIGHTS="):
        return f"训练权重: {line.split('=', 1)[1]}"
    if line.startswith("ONNX="):
        return f"ONNX: {line.split('=', 1)[1]}"
    if line.startswith("DOCKER_COPY="):
        return f"Docker 输入目录: {line.split('=', 1)[1]}"
    if line.startswith("Epoch") and "GPU_mem" in line:
        return ""
    if line.startswith(("github:", "Comet:", "TensorBoard:")):
        return ""
    if line.startswith("from") and "params" in line:
        return ""
    if MODEL_ROW.match(line):
        return ""
    return line


def stream_process(command):
    """运行子进程并整理其标准输出，返回退出码和完整文本。"""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    output = []
    active_width = 0
    active_line = False

    def emit(raw_line, ending):
        nonlocal active_line, active_width
        if raw_line:
            output.append(raw_line)
        formatted = format_output_line(raw_line)
        if formatted == "":
            if ending == "\n" and active_line:
                sys.stdout.write("\n")
                active_line = False
                active_width = 0
            return
        if formatted is None:
            if ending == "\n" and active_line:
                sys.stdout.write("\n")
                active_line = False
                active_width = 0
            if formatted is None and raw_line and "%|" not in raw_line:
                print(raw_line, flush=True)
            return
        padding = max(0, active_width - len(formatted))
        sys.stdout.write("\r" + formatted + " " * padding)
        active_line = True
        active_width = max(active_width, len(formatted))
        if ending == "\n":
            sys.stdout.write("\n")
            active_line = False
            active_width = 0
        sys.stdout.flush()

    while chunk := process.stdout.read(4096):
        text = decoder.decode(chunk)
        for character in text:
            if character in "\r\n":
                emit(pending, character)
                pending = ""
            else:
                pending += character
    pending += decoder.decode(b"", final=True)
    if pending:
        emit(pending, "\n")
    if active_line:
        sys.stdout.write("\n")
        sys.stdout.flush()
    process.wait()
    return process.returncode, "\n".join(output)


def dataset_source(dataset_path):
    """从 YOLO 数据集 YAML 中读取训练图片目录。"""
    dataset = load_config(dataset_path)
    source = dataset.get("train")
    if isinstance(source, list):
        source = source[0] if source else None
    return resolve(source, dataset_path.parent) if source else None


def trained_weights(project, name):
    """查找指定训练任务的 best.pt，找不到时返回最新结果。"""
    expected = resolve(project) / name / "weights" / "best.pt"
    if expected.exists():
        return expected
    candidates = sorted(
        resolve(project).glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else expected


def prepare_docker(config, onnx):
    """将 ONNX 和校准图片复制到 Docker 共享目录。"""
    spec = config.get("docker", {})
    directory_value = spec.get("directory")
    if not directory_value:
        return None
    directory = resolve(directory_value, ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    model_name = spec.get("model_name") or onnx.stem
    shutil.copy2(onnx, directory / f"{model_name}.onnx")

    source = spec.get("source") or dataset_source(resolve(config["dataset"], ROOT))
    if not source:
        return directory
    source = resolve(source, ROOT)
    if not source.exists():
        raise FileNotFoundError(f"sample source does not exist: {source}")
    pool = image_files(source)
    random.Random(0).shuffle(pool)
    sample_count = int(spec.get("sample_count", 10))
    if len(pool) < sample_count:
        raise RuntimeError(
            f"need {sample_count} sample images, "
            f"found {len(pool)} in {source}"
        )
    for index, image in enumerate(pool[:sample_count]):
        shutil.copy2(image, directory / f"test{index}.jpg")
    calibration_dir = directory / "images"
    calibration_dir.mkdir(exist_ok=True)
    for image in pool[: int(spec.get("calibration_count", 200))]:
        shutil.copy2(image, calibration_dir / image.name)
    return directory


def worker_train(config_path):
    """在 .trainenv 环境中执行训练阶段。"""
    config = load_config(config_path)
    base = ROOT
    train_spec = dict(config.get("train", {}))
    dataset = resolve(config["dataset"], base)
    weights = resolve(config["weights"], base)
    if not dataset.exists():
        raise FileNotFoundError(f"dataset config does not exist: {dataset}")
    if not weights.exists():
        raise FileNotFoundError(f"initial weights do not exist: {weights}")
    import torch
    import train

    if torch.cuda.is_available():
        print(f"TRAIN_DEVICE=cuda:0 {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("TRAIN_DEVICE=cpu (CUDA unavailable)", flush=True)

    train.run(data=str(dataset), weights=str(weights), **train_spec)
    result = trained_weights(
        train_spec.get("project", "runs/train"), train_spec.get("name", "maixcam")
    )
    if not result.exists():
        raise FileNotFoundError(
            f"training completed but best.pt was not found: {result}"
        )
    print(f"TRAIN_WEIGHTS={result}", flush=True)


def worker_export(config_path, weights_path):
    """在 .exportenv 环境中执行 MaixCam ONNX 导出阶段。"""
    config = load_config(config_path)
    weights = resolve(weights_path, ROOT)
    if not weights.exists():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    from tools.maixcam_export import export_pure

    spec = config.get("export", {})
    output = (
        resolve(spec.get("project", "runs/export"), ROOT)
        / spec.get("name", "maixcam")
        / f"{weights.stem}.onnx"
    )
    export_pure(
        weights,
        output,
        int(spec.get("width", 256)),
        int(spec.get("height", 160)),
        int(spec.get("opset", 12)),
        spec.get("output_names", ["out0", "out1", "out2"]),
    )
    docker = prepare_docker(config, output)
    print(f"ONNX={output}")
    if docker:
        print(f"DOCKER_COPY={docker}")
        print(
            "在 tpu-env 当前目录执行: chmod +x convert_yolov5_to_cvimodel.sh && ./convert_yolov5_to_cvimodel.sh"
        )


def run_orchestrator(config_path):
    """分别启动训练和导出子进程，并实时转发训练日志。"""
    train_python = ROOT / ".trainenv" / "Scripts" / "python.exe"
    convert_python = ROOT / ".exportenv" / "Scripts" / "python.exe"
    for interpreter in (train_python, convert_python):
        if not interpreter.exists():
            raise FileNotFoundError(
                f"required environment interpreter does not exist: {interpreter}"
            )

    train_command = [
        str(train_python),
        "-u",
        str(SCRIPT),
        "--phase",
        "train",
        "--config",
        str(config_path),
    ]
    print("[1/2] 开始训练（.trainenv）", flush=True)
    train_code, train_output = stream_process(train_command)
    if train_code:
        raise SystemExit(train_code)
    match = re.search(r"^TRAIN_WEIGHTS=(.+)$", train_output, re.MULTILINE)
    if not match:
        raise RuntimeError("training finished but no TRAIN_WEIGHTS marker was returned")

    export_command = [
        str(convert_python),
        "-u",
        str(SCRIPT),
        "--phase",
        "export",
        "--config",
        str(config_path),
        "--weights",
        match.group(1).strip(),
    ]
    print("[2/2] 开始导出 MaixCam ONNX（.exportenv）", flush=True)
    export_code, _ = stream_process(export_command)
    if export_code == 0:
        print("[完成] 训练、导出和 Docker 输入准备已完成。", flush=True)
    raise SystemExit(export_code)


def main():
    """解析命令行参数并分派工作流阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "maixcam.yaml"
    )
    parser.add_argument("--phase", choices=("run", "train", "export"), default="run")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        run_orchestrator(args.config.resolve())
    elif args.phase == "train":
        worker_train(args.config.resolve())
    else:
        if not args.weights:
            raise ValueError("--weights is required for --phase export")
        worker_export(args.config.resolve(), args.weights)


if __name__ == "__main__":
    main()
