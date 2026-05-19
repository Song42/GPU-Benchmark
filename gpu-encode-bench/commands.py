"""Command builders and deterministic naming helpers."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from bench_config import (
    AXIS_BITRATE,
    AXIS_CQ23,
    CODEC_ENCODERS,
    CQ_VALUE,
    PRESET,
    bitrate_bufsize_mbps,
)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "gpu"


def mode_label(case: dict[str, Any]) -> str:
    if case["benchmark_axis"] == AXIS_BITRATE:
        bitrate = case["target_bitrate_mbps"]
        if float(bitrate).is_integer():
            bitrate_text = str(int(bitrate))
        else:
            bitrate_text = str(bitrate).replace(".", "p")
        return f"{bitrate_text}m"
    return "cq23"


def output_filename(
    gpu_slug: str,
    case: dict[str, Any],
    parallel_jobs: int,
    job_index: int,
) -> str:
    return (
        f"{gpu_slug}_{case['benchmark_axis']}_{case['source_label']}_"
        f"{case['codec_label']}_{mode_label(case)}_{parallel_jobs}x_{job_index}.mp4"
    )


def failed_log_filename(
    gpu_slug: str,
    case: dict[str, Any],
    parallel_jobs: int,
    job_index: int,
) -> str:
    return output_filename(gpu_slug, case, parallel_jobs, job_index).replace(
        ".mp4", "_FAILED.log"
    )


def build_encode_command(
    case: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    encoder = CODEC_ENCODERS[case["codec_label"]]
    if case["benchmark_axis"] == AXIS_BITRATE:
        bitrate = case["target_bitrate_mbps"]
        bufsize = bitrate_bufsize_mbps(bitrate)
        return [
            "ffmpeg",
            "-benchmark",
            "-y",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-c:v",
            encoder,
            "-preset",
            PRESET,
            "-rc",
            "cbr",
            "-b:v",
            f"{bitrate}M",
            "-maxrate",
            f"{bitrate}M",
            "-bufsize",
            f"{bufsize}M",
            "-an",
            str(output_path),
        ]

    return [
        "ffmpeg",
        "-benchmark",
        "-y",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-c:v",
        encoder,
        "-preset",
        PRESET,
        "-rc",
        "vbr",
        "-cq",
        str(CQ_VALUE),
        "-b:v",
        "0",
        "-an",
        str(output_path),
    ]


def build_vmaf_command(
    encoded_path: str | Path,
    source_path: str | Path,
    vmaf_json_path: str | Path,
) -> list[str]:
    filter_expr = (
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS[dist];"
        "[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];"
        "[dist][ref]libvmaf="
        f"log_fmt=json:log_path={vmaf_json_path}:model=version=vmaf_v0.6.1"
    )
    return [
        "ffmpeg",
        "-i",
        str(encoded_path),
        "-i",
        str(source_path),
        "-lavfi",
        filter_expr,
        "-f",
        "null",
        "-",
    ]


def command_to_string(command: list[str]) -> str:
    return shlex.join(command)
