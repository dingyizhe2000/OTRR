"""System and GPU monitoring for simulation runs."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_PROGRESS_FILE = None

_GPU_MONITOR_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
    "memory.total",
    "power.draw",
    "power.limit",
    "enforced.power.limit",
    "temperature.gpu",
    "temperature.memory",
    "fan.speed",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.current.video",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.gpu_idle",
    "clocks_throttle_reasons.applications_clocks_setting",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sync_boost",
    "voltage.graphics",
]


def _monitor_column(field: str) -> str:
    return "gpu_" + re.sub(r"[^A-Za-z0-9]+", "_", field).strip("_")


def _read_proc_stat() -> tuple[int, int] | None:
    try:
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    except OSError:
        return None
    values = [int(value) for value in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_percent(prev: tuple[int, int] | None, cur: tuple[int, int] | None) -> float | None:
    if prev is None or cur is None:
        return None
    total_delta = cur[0] - prev[0]
    idle_delta = cur[1] - prev[1]
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        out[key] = value * 1024 if len(parts) > 1 and parts[1].lower() == "kb" else value
    return out


def _loadavg() -> tuple[float | None, float | None, float | None]:
    try:
        values = os.getloadavg()
    except OSError:
        return None, None, None
    return float(values[0]), float(values[1]), float(values[2])


def _detect_gpu_monitor_fields() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--help-query-gpu"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    help_text = result.stdout
    fields = [field for field in _GPU_MONITOR_FIELDS if field in help_text]
    # timestamp is accepted by nvidia-smi even though some help versions do not
    # list it with the same formatting.
    if "timestamp" not in fields:
        fields.insert(0, "timestamp")
    return fields


def _query_gpu_monitor(fields: list[str]) -> tuple[list[dict[str, str]], str | None]:
    if not fields:
        return [], "nvidia-smi unavailable or no supported query fields detected"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], repr(exc)
    if result.returncode != 0:
        return [], result.stderr.strip() or result.stdout.strip() or f"nvidia-smi exited {result.returncode}"
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        rows.append({field: values[index] if index < len(values) else "" for index, field in enumerate(fields)})
    return rows, None


def _monitor_header() -> list[str]:
    return [
        "wall_time",
        "monotonic_sec",
        "load1",
        "load5",
        "load15",
        "cpu_percent",
        "mem_total_bytes",
        "mem_available_bytes",
        "swap_total_bytes",
        "swap_free_bytes",
        "nvidia_smi_error",
    ] + [_monitor_column(field) for field in _GPU_MONITOR_FIELDS]


def _monitor_loop(
    *,
    path: Path,
    error_path: Path,
    interval_sec: float,
    stop_event: threading.Event,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _detect_gpu_monitor_fields()
    header = _monitor_header()
    start = time.monotonic()
    prev_cpu = _read_proc_stat()
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=header).writeheader()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        while not stop_event.is_set():
            cur_cpu = _read_proc_stat()
            mem = _read_meminfo()
            load1, load5, load15 = _loadavg()
            base = {
                "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "monotonic_sec": f"{time.monotonic() - start:.3f}",
                "load1": load1,
                "load5": load5,
                "load15": load15,
                "cpu_percent": None if (cpu := _cpu_percent(prev_cpu, cur_cpu)) is None else f"{cpu:.3f}",
                "mem_total_bytes": mem.get("MemTotal"),
                "mem_available_bytes": mem.get("MemAvailable"),
                "swap_total_bytes": mem.get("SwapTotal"),
                "swap_free_bytes": mem.get("SwapFree"),
                "nvidia_smi_error": "",
            }
            prev_cpu = cur_cpu
            gpu_rows, error = _query_gpu_monitor(fields)
            if error:
                error_path.parent.mkdir(parents=True, exist_ok=True)
                with error_path.open("a") as err:
                    err.write(json.dumps({"time": time.time(), "error": error}) + "\n")
                row = dict(base)
                row["nvidia_smi_error"] = error
                writer.writerow(row)
            elif gpu_rows:
                for gpu_row in gpu_rows:
                    row = dict(base)
                    for field in _GPU_MONITOR_FIELDS:
                        row[_monitor_column(field)] = gpu_row.get(field, "")
                    writer.writerow(row)
            else:
                writer.writerow(base)
            handle.flush()
            stop_event.wait(max(0.5, float(interval_sec)))


class _SimulationMonitor:
    def __init__(self, out_dir: Path, runtime_cfg: dict[str, Any]):
        cfg = runtime_cfg.get("system_monitor", {})
        if isinstance(cfg, dict):
            enabled = cfg.get("enabled", runtime_cfg.get("system_monitor_enabled", True))
            interval = cfg.get("interval_sec", runtime_cfg.get("system_monitor_interval_sec", 5))
            filename = cfg.get("filename", runtime_cfg.get("system_monitor_filename", "system_monitor.csv"))
        else:
            enabled = bool(cfg)
            interval = runtime_cfg.get("system_monitor_interval_sec", 5)
            filename = runtime_cfg.get("system_monitor_filename", "system_monitor.csv")
        self.enabled = bool(enabled)
        self.path = out_dir / str(filename)
        self.error_path = out_dir / "system_monitor_errors.jsonl"
        self.interval_sec = float(interval)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(
                target=_monitor_loop,
                kwargs={
                    "path": self.path,
                    "error_path": self.error_path,
                    "interval_sec": self.interval_sec,
                    "stop_event": self.stop_event,
                },
                daemon=True,
            )
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=max(2.0, self.interval_sec + 2.0))
        return False

def _progress_file():
    """Write progress bars to the controlling terminal when stdout/stderr are captured."""
    global _PROGRESS_FILE
    if _PROGRESS_FILE is not None:
        return _PROGRESS_FILE
    try:
        _PROGRESS_FILE = open("/dev/tty", "w", buffering=1)
    except OSError:
        _PROGRESS_FILE = sys.stderr
    return _PROGRESS_FILE
