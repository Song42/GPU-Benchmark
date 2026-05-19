"""Parsing helpers for ffmpeg, ffprobe, nvidia-smi, and VMAF outputs."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    cleaned = str(value).strip().replace(",", "")
    cleaned = re.sub(r"\s*(MiB|MB|W|C|MHz|%)$", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def mb_from_bytes(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    return number / (1024 * 1024)


def mbps_from_bps(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    return number / 1_000_000


def parse_ffprobe_json(text: str) -> dict[str, Any]:
    data = json.loads(text or "{}")
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}
    duration = safe_float(stream.get("duration")) or safe_float(fmt.get("duration"))
    bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
    return {
        "width": safe_int(stream.get("width")),
        "height": safe_int(stream.get("height")),
        "fps": parse_fraction(stream.get("avg_frame_rate"))
        or parse_fraction(stream.get("r_frame_rate")),
        "duration_sec": duration,
        "video_codec": stream.get("codec_name") or "",
        "pix_fmt": stream.get("pix_fmt") or "",
        "video_bitrate_mbps": mbps_from_bps(bitrate),
        "file_size_mb": mb_from_bytes(fmt.get("size")),
    }


def ffprobe_media(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,codec_name,pix_fmt,bit_rate",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.strip()
    try:
        return parse_ffprobe_json(completed.stdout), None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, str(exc)


def parse_ffmpeg_benchmark_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ffmpeg_rtime_sec": None,
        "ffmpeg_utime_sec": None,
        "ffmpeg_stime_sec": None,
        "speed_x": None,
        "frames_encoded": None,
        "avg_fps": None,
    }
    bench_match = re.search(
        r"bench:\s+utime=(?P<utime>[0-9.]+)s\s+stime=(?P<stime>[0-9.]+)s\s+rtime=(?P<rtime>[0-9.]+)s",
        text,
    )
    if bench_match:
        result["ffmpeg_utime_sec"] = safe_float(bench_match.group("utime"))
        result["ffmpeg_stime_sec"] = safe_float(bench_match.group("stime"))
        result["ffmpeg_rtime_sec"] = safe_float(bench_match.group("rtime"))

    speed_matches = re.findall(r"speed=\s*([0-9.]+)x", text)
    if speed_matches:
        result["speed_x"] = safe_float(speed_matches[-1])

    frame_matches = re.findall(r"frame=\s*([0-9]+)", text)
    if frame_matches:
        result["frames_encoded"] = safe_int(frame_matches[-1])

    fps_matches = re.findall(r"fps=\s*([0-9.]+)", text)
    if fps_matches:
        result["avg_fps"] = safe_float(fps_matches[-1])

    return result


def parse_vmaf_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pooled = data.get("pooled_metrics", {}).get("vmaf", {})
    frames = data.get("frames") or []
    frame_scores = []
    for frame in frames:
        score = frame.get("metrics", {}).get("vmaf")
        if score is not None:
            frame_scores.append(float(score))

    scores_sorted = sorted(frame_scores)

    def percentile(percent: float) -> float | None:
        if not scores_sorted:
            return None
        if len(scores_sorted) == 1:
            return scores_sorted[0]
        position = (len(scores_sorted) - 1) * percent
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return scores_sorted[lower]
        weight = position - lower
        return scores_sorted[lower] * (1 - weight) + scores_sorted[upper] * weight

    return {
        "vmaf_mean": safe_float(pooled.get("mean"))
        or (statistics.fmean(frame_scores) if frame_scores else None),
        "vmaf_min": safe_float(pooled.get("min"))
        or (min(frame_scores) if frame_scores else None),
        "vmaf_max": safe_float(pooled.get("max"))
        or (max(frame_scores) if frame_scores else None),
        "vmaf_harmonic_mean": safe_float(pooled.get("harmonic_mean")),
        "vmaf_p1": safe_float(pooled.get("perc1")) or percentile(0.01),
        "vmaf_p5": safe_float(pooled.get("perc5")) or percentile(0.05),
        "vmaf_p95": safe_float(pooled.get("perc95")) or percentile(0.95),
        "vmaf_p99": safe_float(pooled.get("perc99")) or percentile(0.99),
    }


def clean_nvidia_value(value: str) -> str:
    return value.strip().replace(" [Not Supported]", "").replace("[N/A]", "").strip()


def parse_gpu_monitor_log(
    raw_path: str | Path,
    run_id: str,
    batch_id: str,
    parallel_jobs: int,
) -> list[dict[str, Any]]:
    path = Path(raw_path)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    headers: list[str] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = [clean_nvidia_value(part) for part in line.split(",")]
        if line.lower().startswith("timestamp"):
            headers = parts
            continue
        if headers and len(parts) < len(headers):
            continue
        if len(parts) < 13:
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "gpu_name": parts[1],
                "gpu_index": parts[2],
                "util_gpu_percent": safe_float(parts[3]),
                "util_memory_percent": safe_float(parts[4]),
                "util_encoder_percent": safe_float(parts[5]),
                "util_decoder_percent": safe_float(parts[6]),
                "memory_used_mb": safe_float(parts[7]),
                "memory_total_mb": safe_float(parts[8]),
                "power_draw_w": safe_float(parts[9]),
                "temperature_c": safe_float(parts[10]),
                "clocks_sm_mhz": safe_float(parts[11]),
                "clocks_mem_mhz": safe_float(parts[12]),
                "run_id": run_id,
                "batch_id": batch_id,
                "parallel_jobs": parallel_jobs,
            }
        )
    return rows


def average(values: list[float | int | None]) -> float | None:
    clean = []
    for value in values:
        number = safe_float(value)
        if number is not None:
            clean.append(number)
    if not clean:
        return None
    return statistics.fmean(clean)


def maximum(values: list[float | int | None]) -> float | None:
    clean = []
    for value in values:
        number = safe_float(value)
        if number is not None:
            clean.append(number)
    if not clean:
        return None
    return max(clean)
