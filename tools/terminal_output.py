"""解析 YOLOv5 日志并显示训练进度与指标。"""

from __future__ import annotations

import re
import shutil
import sys
import time
import unicodedata
from typing import TextIO

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_ROW = re.compile(
    rf"^(\d+)/(\d+)\s+(\S+)\s+({NUMBER})\s+({NUMBER})\s+"
    rf"({NUMBER})\s+(\d+)\s+(\d+):\s*(\d+)%\|"
)
METRIC_ROW = re.compile(
    rf"^(.+?)\s+(\d+)\s+(\d+)\s+({NUMBER})\s+({NUMBER})\s+"
    rf"({NUMBER})\s+({NUMBER})$"
)
MODEL_ROW = re.compile(r"^\d+\s+(?:-?\d+|\[[^]]+\])\s+\d+\s+\d+\s+models\.")
PROFILE_ROW = re.compile(r"^\d+\s+[\d.]+\s+[\d.]+.*\(.*\)")
BAR = re.compile(r"(\d+)%\|.*?\|\s*(\d+/\d+)")


def cell_width(text: str) -> int:
    """计算包含中文字符的终端显示宽度。"""
    return sum(
        0 if unicodedata.combining(char)
        else 2 if unicodedata.east_asian_width(char) in "WF" else 1
        for char in text
    )


def progress_bar(percent: float, width: int = 18) -> str:
    """生成指定宽度的训练总进度条。"""
    filled = int(width * min(100, max(0, percent)) / 100)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def duration(seconds: float) -> str:
    """将秒数转换为分秒文本。"""
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


class WorkflowOutput:
    """合并批次刷新记录，并在验证结束后输出每轮指标。"""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = self.stream.isatty()
        self.pending = ""
        self.live_width = 0
        self.epoch = None
        self.total = 0
        self.completed = 0
        self.reported = set()
        self.scans = set()
        self.final_validation = False
        self.table_printed = False
        self.started = None
        self.last_refresh = 0.0
        self.traceback = False

    def clear_live(self) -> None:
        """清除终端中的动态状态行。"""
        if self.live_width:
            self.stream.write("\r" + " " * self.live_width + "\r")
            self.live_width = 0

    def write(self, text: str = "") -> None:
        """输出一条永久保留的日志。"""
        self.clear_live()
        self.stream.write(text + "\n")
        self.stream.flush()

    def live(self, text: str) -> None:
        """按终端宽度刷新动态状态。"""
        if not self.interactive:
            return
        now = time.monotonic()
        if now - self.last_refresh < 0.1:
            return
        self.last_refresh = now
        self.clear_live()
        limit = shutil.get_terminal_size((160, 24)).columns - 1
        visible = ""
        for char in text:
            if cell_width(visible + char) > limit:
                break
            visible += char
        self.stream.write("\r" + visible)
        self.stream.flush()
        self.live_width = cell_width(visible)

    def feed(self, text: str) -> None:
        """接收任意分块的文本，按回车或换行解析刷新记录。"""
        records = re.split(r"[\r\n]", self.pending + text)
        self.pending = records.pop()
        for record in records:
            if record.strip():
                self.consume(record)

    def finish(self) -> None:
        """输出末尾记录并清除动态状态。"""
        if self.pending:
            self.consume(self.pending)
            self.pending = ""
        self.clear_live()
        self.stream.flush()

    def summary(self, line: str) -> None:
        """将训练参数展开为路径与运行配置摘要。"""
        fields = dict(
            part.split("=", 1)
            for part in re.split(r", (?=\w+=)", line[7:])
            if "=" in part
        )
        self.write("\n训练配置")
        self.write(f"  数据集  {fields.get('data', '-')}")
        self.write(f"  权重    {fields.get('weights', '-')}")
        batch = fields.get("batch_size", "-")
        batch = "自动" if batch == "-1" else batch
        self.write(
            f"  轮数 {fields.get('epochs', '-')}    "
            f"输入 {fields.get('imgsz', '-')} px    Batch {batch}    "
            f"Workers {fields.get('workers', '-')}    "
            f"缓存 {fields.get('cache', '-')}"
        )

    def training_table(self) -> None:
        """输出每轮训练指标的列名。"""
        if self.table_printed:
            return
        self.table_printed = True
        self.write("\n训练指标（总进度在每轮验证完成后更新）")
        self.write(
            f"{'Epoch':>9}  {'Total progress':<27}  {'GPU_mem':>7}  "
            f"{'box_loss':>10} {'obj_loss':>10} {'cls_loss':>10}  "
            f"{'Precision':>9} {'Recall':>9} {'mAP50':>9} "
            f"{'mAP50-95':>9}  {'Elapsed':>7}"
        )
        self.write("-" * 143)

    def metrics(self, match: re.Match) -> None:
        """输出每轮验证指标或最终模型的分类指标。"""
        name, images, instances, *values = match.groups()
        numbers = " ".join(f"{float(v):9.4g}" for v in values)
        if self.epoch and name == "all" and not self.final_validation:
            index, _, gpu, box, obj, cls, _, _, _ = self.epoch
            if index in self.reported:
                return
            self.reported.add(index)
            self.completed = int(index) + 1
            percent = self.completed / self.total * 100
            elapsed = time.monotonic() - self.started
            self.training_table()
            self.write(
                f"{self.completed:>4}/{self.total:<4}  "
                f"{progress_bar(percent)} {percent:5.1f}%   {gpu:>7}  "
                f"{float(box):10.4g} {float(obj):10.4g} "
                f"{float(cls):10.4g}  {numbers}  {duration(elapsed):>7}"
            )
            remaining = elapsed / len(self.reported) * (
                self.total - self.completed
            )
            self.live(
                f"总进度 {progress_bar(percent, 32)} {percent:5.1f}%  "
                f"Epoch {self.completed}/{self.total}  "
                f"已用 {duration(elapsed)}  预计剩余 {duration(remaining)}"
            )
        else:
            self.write(
                f"  {name:<18} Images {images:>5}  "
                f"Instances {instances:>5}  {numbers}"
            )

    def consume(self, raw: str) -> None:
        """识别进度、配置、指标和诊断日志。"""
        line = ANSI_ESCAPE.sub("", raw).strip()
        if self.traceback or line.startswith("Traceback"):
            self.traceback = True
            self.write(raw)
            return
        if re.search(r"warning|error|exception", line, re.IGNORECASE):
            if "0 WARNING 0 ERROR" not in line:
                self.write(raw)
                return
        match = TRAIN_ROW.match(line)
        if match:
            self.epoch = match.groups()
            index, last, gpu, *_, batch_percent = self.epoch
            self.total = int(last) + 1
            if self.started is None:
                self.started = time.monotonic()
            self.training_table()
            percent = self.completed / self.total * 100
            self.live(
                f"总进度 {progress_bar(percent, 32)} {percent:5.1f}%  "
                f"Epoch {int(index) + 1}/{self.total}  "
                f"训练批次 {batch_percent}%  GPU_mem {gpu}"
            )
            return
        match = METRIC_ROW.match(line)
        if match:
            self.metrics(match)
            return
        match = BAR.search(line)
        if match:
            percent, count = match.groups()
            if "Scanning" in line or "Caching" in line:
                split = "train" if line.startswith("train:") else "val"
                label = "扫描" if "Scanning" in line else "缓存"
                key = (split, label)
                detail = line.split(":", 1)[-1].split("%|", 1)[0]
                self.live(f"数据 {split} · {label}  {percent}%  {count}")
                if percent == "100" and key not in self.scans:
                    self.scans.add(key)
                    counts = re.search(
                        r"(\d+ images, .*?corrupt)", detail
                    )
                    self.write(
                        f"  数据 {split} · {label}完成  "
                        f"{counts.group(1) if counts else count}"
                    )
            elif "Class" in line or "mAP" in line:
                label = "最终验证" if self.final_validation else "验证批次"
                self.live(f"{label} {progress_bar(int(percent), 32)} "
                          f"{percent}%  {count}")
            else:
                self.live(f"准备 {progress_bar(int(percent), 32)} "
                          f"{percent}%  {count}")
            return
        if line.startswith("train: weights="):
            self.summary(line)
        elif line.startswith("TRAIN_DEVICE="):
            self.write(f"  设备    {line.split('=', 1)[1]}")
        elif line.startswith("Logging results to "):
            self.write(f"  训练目录  {line[19:]}")
        elif line.startswith("Starting training for "):
            self.started = time.monotonic()
            self.write(f"\n开始训练  {line[22:]}")
        elif line.startswith("Validating "):
            self.final_validation = True
            self.write(f"\n最终权重验证  {line[11:]}")
            self.write(
                "  Class                  Images / Instances    "
                "Precision    Recall     mAP50  mAP50-95"
            )
        elif line.startswith(("TRAIN_WEIGHTS=", "ONNX=", "DOCKER_COPY=")):
            key, value = line.split("=", 1)
            label = {"TRAIN_WEIGHTS": "训练权重", "ONNX": "ONNX",
                     "DOCKER_COPY": "Docker 输入目录"}[key]
            self.write(f"  {label}  {value}")
        elif line.startswith("AutoBatch: Using batch-size "):
            self.write(f"  Batch   {line[28:]}")
        elif (
            line.startswith(("hyperparameters:", "github:", "Comet:",
                             "TensorBoard:", "AutoBatch: Computing",
                             "AutoBatch: CUDA:", "=============="))
            or (line.startswith("Epoch") and "GPU_mem" in line)
            or (line.startswith("from") and "params" in line)
            or (line.startswith("Params") and "GFLOPs" in line)
            or MODEL_ROW.match(line)
            or PROFILE_ROW.match(line)
            or line == "verbose: False, log level: Level.ERROR"
        ):
            return
        elif line:
            self.write(raw)
