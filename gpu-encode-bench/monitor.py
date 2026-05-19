"""nvidia-smi monitoring lifecycle helpers."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from parser import average, maximum, parse_gpu_monitor_log


@dataclass
class MonitorHandle:
    process: subprocess.Popen | None
    raw_path: Path
    log_file: object | None
    started: bool
    error: str = ""


MONITOR_QUERY = (
    "timestamp,name,index,utilization.gpu,utilization.memory,"
    "utilization.encoder,utilization.decoder,memory.used,memory.total,"
    "power.draw,temperature.gpu,clocks.sm,clocks.mem"
)


def start_monitor(
    log_dir: Path,
    batch_id: str,
    parallel_jobs: int,
) -> MonitorHandle:
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_path = log_dir / f"{batch_id}_{parallel_jobs}x_nvidia_smi.csv"
    log_file = raw_path.open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        f"--query-gpu={MONITOR_QUERY}",
        "--format=csv",
        "-l",
        "1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        log_file.write(str(exc))
        log_file.close()
        return MonitorHandle(None, raw_path, None, False, str(exc))
    return MonitorHandle(process, raw_path, log_file, True)


def stop_monitor(handle: MonitorHandle) -> None:
    if handle.process is not None and handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=5)
    if handle.log_file is not None:
        time.sleep(0.1)
        handle.log_file.close()


def parse_monitor_rows(
    handle: MonitorHandle,
    run_id: str,
    batch_id: str,
    parallel_jobs: int,
) -> list[dict]:
    return parse_gpu_monitor_log(handle.raw_path, run_id, batch_id, parallel_jobs)


def monitor_aggregates(rows: list[dict]) -> dict[str, float | None]:
    return {
        "gpu_util_avg_percent": average([row.get("util_gpu_percent") for row in rows]),
        "gpu_util_max_percent": maximum([row.get("util_gpu_percent") for row in rows]),
        "encoder_util_avg_percent": average(
            [row.get("util_encoder_percent") for row in rows]
        ),
        "encoder_util_max_percent": maximum(
            [row.get("util_encoder_percent") for row in rows]
        ),
        "decoder_util_avg_percent": average(
            [row.get("util_decoder_percent") for row in rows]
        ),
        "memory_used_avg_mb": average([row.get("memory_used_mb") for row in rows]),
        "memory_used_max_mb": maximum([row.get("memory_used_mb") for row in rows]),
        "power_draw_avg_w": average([row.get("power_draw_w") for row in rows]),
        "temperature_avg_c": average([row.get("temperature_c") for row in rows]),
    }
