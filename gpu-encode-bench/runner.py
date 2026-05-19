"""Execution helpers for encode batches and VMAF commands."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_config import (
    STATUS_FAIL_FFMPEG_ERROR,
    STATUS_FAIL_OUTPUT_MISSING,
    STATUS_FAIL_TIMEOUT,
    STATUS_FAIL_UNSUPPORTED_ENCODER,
    STATUS_SUCCESS,
)
from parser import ffprobe_media, parse_ffmpeg_benchmark_output


@dataclass
class JobSpec:
    job_id: str
    job_index: int
    command: list[str]
    output_path: Path
    log_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_ffmpeg_error(text: str, encoder: str) -> tuple[str, str, str]:
    lowered = text.lower()
    if (
        "unknown encoder" in lowered
        or encoder.lower() in lowered and "not recognized" in lowered
        or "specified rc mode is not supported" in lowered
        or "no capable devices found" in lowered
        or "device does not support" in lowered
    ):
        return STATUS_FAIL_UNSUPPORTED_ENCODER, "unsupported_encoder", first_error_line(text)
    if (
        "too many concurrent sessions" in lowered
        or "openencodesessionex failed" in lowered
        or "nv_enc_err" in lowered
    ):
        return STATUS_FAIL_FFMPEG_ERROR, "nvenc_session", first_error_line(text)
    return STATUS_FAIL_FFMPEG_ERROR, "ffmpeg_error", first_error_line(text)


def first_error_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        clean = line.strip()
        if clean:
            return clean[:500]
    return ""


def run_parallel_jobs(
    job_specs: list[JobSpec],
    timeout_sec: int,
    encoder: str,
) -> tuple[list[dict[str, Any]], float]:
    processes: list[dict[str, Any]] = []
    batch_start = time.monotonic()
    for spec in job_specs:
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = spec.log_path.open("w", encoding="utf-8")
        log_file.write("# command: " + " ".join(spec.command) + "\n")
        log_file.flush()
        timestamp_start = utc_now()
        start_time = time.monotonic()
        try:
            process = subprocess.Popen(
                spec.command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(
                {
                    "spec": spec,
                    "process": process,
                    "log_file": log_file,
                    "start_time": start_time,
                    "end_time": None,
                    "timestamp_start": timestamp_start,
                    "timestamp_end": "",
                    "timed_out": False,
                }
            )
        except OSError as exc:
            log_file.write(str(exc))
            log_file.close()
            processes.append(
                {
                    "spec": spec,
                    "process": None,
                    "log_file": None,
                    "start_time": start_time,
                    "end_time": time.monotonic(),
                    "timestamp_start": timestamp_start,
                    "timestamp_end": utc_now(),
                    "spawn_error": str(exc),
                    "timed_out": False,
                }
            )

    while True:
        all_done = True
        now = time.monotonic()
        timed_out = now - batch_start > timeout_sec
        for entry in processes:
            process = entry.get("process")
            if process is None or entry.get("end_time") is not None:
                continue
            if timed_out and process.poll() is None:
                entry["timed_out"] = True
                process.kill()
            if process.poll() is None:
                all_done = False
            else:
                process.wait()
                entry["end_time"] = time.monotonic()
                entry["timestamp_end"] = utc_now()
                if entry.get("log_file") is not None:
                    entry["log_file"].flush()
                    entry["log_file"].close()
                    entry["log_file"] = None
        if all_done:
            break
        time.sleep(0.25)

    batch_time = time.monotonic() - batch_start
    results = []
    for entry in processes:
        if entry.get("log_file") is not None:
            entry["log_file"].close()
        spec = entry["spec"]
        end_time = entry.get("end_time") or time.monotonic()
        log_text = ""
        if spec.log_path.exists():
            log_text = spec.log_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_ffmpeg_benchmark_output(log_text)
        process = entry.get("process")
        exit_code = process.returncode if process is not None else None
        status = STATUS_SUCCESS
        error_type = ""
        error_message = ""
        output_metadata = None

        if entry.get("spawn_error"):
            status = STATUS_FAIL_FFMPEG_ERROR
            error_type = "spawn_error"
            error_message = entry["spawn_error"]
        elif entry.get("timed_out"):
            status = STATUS_FAIL_TIMEOUT
            error_type = "timeout"
            error_message = f"Exceeded batch timeout of {timeout_sec}s"
        elif exit_code != 0:
            status, error_type, error_message = classify_ffmpeg_error(log_text, encoder)
        else:
            if not spec.output_path.exists() or spec.output_path.stat().st_size <= 0:
                status = STATUS_FAIL_OUTPUT_MISSING
                error_type = "output_missing"
                error_message = "Output file missing or empty"
            else:
                output_metadata, probe_error = ffprobe_media(spec.output_path)
                if output_metadata is None:
                    status = STATUS_FAIL_OUTPUT_MISSING
                    error_type = "output_probe_failed"
                    error_message = probe_error or "ffprobe could not read output"

        results.append(
            {
                "job_id": spec.job_id,
                "parallel_job_index": spec.job_index,
                "timestamp_start": entry.get("timestamp_start", ""),
                "timestamp_end": entry.get("timestamp_end") or utc_now(),
                "status": status,
                "exit_code": exit_code,
                "error_type": error_type,
                "error_message": error_message,
                "wall_time_sec": end_time - entry["start_time"],
                "batch_time_sec": batch_time,
                "output_exists": spec.output_path.exists(),
                "output_file_size_mb": (
                    spec.output_path.stat().st_size / (1024 * 1024)
                    if spec.output_path.exists()
                    else None
                ),
                "output_metadata": output_metadata or {},
                **parsed,
            }
        )
    return results, batch_time


def run_vmaf_command(
    command: list[str],
    timeout_sec: int,
    log_path: Path,
) -> tuple[bool, int | None, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("# command: " + " ".join(command) + "\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, None, f"VMAF timed out after {timeout_sec}s"
        except OSError as exc:
            return False, None, str(exc)
    if completed.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return False, completed.returncode, first_error_line(text)
    return True, completed.returncode, ""
