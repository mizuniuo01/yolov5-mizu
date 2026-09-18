"""Run the MaixCam train/export workflow from VSCode's Run Code action."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path):
    import yaml

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_path"] = path.resolve()
    return config


def resolve(value, base=ROOT):
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def image_files(path):
    return sorted(
        p
        for p in path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def dataset_source(dataset_path):
    dataset = load_config(dataset_path)
    source = dataset.get("train")
    if isinstance(source, list):
        source = source[0] if source else None
    return resolve(source, dataset_path.parent) if source else None


def trained_weights(project, name):
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
            f"need {sample_count} sample images, found {len(pool)} in {source}"
        )
    for index, image in enumerate(pool[:sample_count]):
        shutil.copy2(image, directory / f"test{index}.jpg")
    calibration_dir = directory / "images"
    calibration_dir.mkdir(exist_ok=True)
    for image in pool[: int(spec.get("calibration_count", 200))]:
        shutil.copy2(image, calibration_dir / image.name)
    return directory


def worker_train(config_path):
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
    train_process = subprocess.Popen(
        train_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    train_lines = []
    assert train_process.stdout is not None
    for line in train_process.stdout:
        print(line, end="", flush=True)
        train_lines.append(line)
    train_process.wait()
    train_output = "".join(train_lines)
    if train_process.returncode:
        raise SystemExit(train_process.returncode)
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
    export_result = subprocess.run(export_command, cwd=ROOT, text=True)
    raise SystemExit(export_result.returncode)


def main():
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
