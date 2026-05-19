"""Preflight checks and environment metadata collection."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_config import CODEC_ENCODERS, FFMPEG_PATH, FFPROBE_PATH, PRESET
from parser import ffprobe_media, mb_from_bytes, safe_int


def run_capture(command: list[str], timeout: int = 60) -> tuple[int | None, str, str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "", str(exc)
    return completed.returncode, completed.stdout, completed.stderr


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ffmpeg_version_summary(ffmpeg_version_text: str) -> tuple[str, str]:
    lines = ffmpeg_version_text.splitlines()
    version = lines[0] if lines else ""
    build_config = ""
    for line in lines:
        if line.startswith("configuration:"):
            build_config = line.removeprefix("configuration:").strip()
            break
    return version, build_config


def parse_gpu_info_csv(text: str) -> list[dict[str, Any]]:
    rows = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return rows
    headers = [header.strip() for header in lines[0].split(",")]
    for index, line in enumerate(lines[1:]):
        parts = [part.strip() for part in line.split(",")]
        row = dict(zip(headers, parts))
        row["index"] = index
        memory_key = "memory.total [MiB]"
        if memory_key in row:
            row["memory_total_mb"] = safe_int(row[memory_key].replace(" MiB", ""))
        rows.append(row)
    return rows


def check_encoder_preset(help_text: str, preset: str = PRESET) -> bool:
    return preset in help_text


def collect_environment(
    env_dir: Path,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    env_dir.mkdir(parents=True, exist_ok=True)
    missing_tools = [
        name
        for name in (FFMPEG_PATH, FFPROBE_PATH, "nvidia-smi")
        if shutil.which(name) is None
    ]

    ffmpeg_rc, ffmpeg_out, ffmpeg_err = run_capture([FFMPEG_PATH, "-version"])
    ffmpeg_text = ffmpeg_out or ffmpeg_err
    write_text(env_dir / "ffmpeg_version.txt", ffmpeg_text)

    encoders_rc, encoders_out, encoders_err = run_capture(
        [FFMPEG_PATH, "-hide_banner", "-encoders"]
    )
    encoders_text = encoders_out or encoders_err
    write_text(env_dir / "ffmpeg_encoders.txt", encoders_text)

    gpu_rc, gpu_out, gpu_err = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,pci.bus_id",
            "--format=csv",
        ]
    )
    gpu_text = gpu_out or gpu_err
    write_text(env_dir / "gpu_info.csv", gpu_text)

    encoder_checks: dict[str, dict[str, Any]] = {}
    for codec_label, encoder in CODEC_ENCODERS.items():
        available = encoder in encoders_text
        help_text = ""
        preset_supported = False
        if available:
            _, help_out, help_err = run_capture(
                [FFMPEG_PATH, "-hide_banner", "-h", f"encoder={encoder}"]
            )
            help_text = help_out or help_err
            write_text(env_dir / f"{encoder}_help.txt", help_text)
            preset_supported = check_encoder_preset(help_text)
        encoder_checks[codec_label] = {
            "encoder": encoder,
            "available": available,
            "preset_slow_supported": preset_supported,
        }

    sources: dict[str, Any] = {}
    for source_label, source_path in source_paths.items():
        if not source_path.exists():
            raise FileNotFoundError(f"{source_label} source not found: {source_path}")
        metadata, error = ffprobe_media(source_path)
        if metadata is None:
            raise RuntimeError(f"ffprobe failed for {source_path}: {error}")
        stat = source_path.stat()
        metadata["file_size_mb"] = metadata.get("file_size_mb") or mb_from_bytes(
            stat.st_size
        )
        metadata["path"] = str(source_path)
        metadata["filename"] = source_path.name
        sources[source_label] = metadata

    ffmpeg_version, ffmpeg_build_config = ffmpeg_version_summary(ffmpeg_text)
    gpu_rows = parse_gpu_info_csv(gpu_text)
    first_gpu = gpu_rows[0] if gpu_rows else {}

    environment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_name": socket.gethostname(),
        "os_name": platform.system(),
        "os_version": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "missing_tools": missing_tools,
        "ffmpeg_path": FFMPEG_PATH,
        "ffprobe_path": FFPROBE_PATH,
        "ffmpeg_version": ffmpeg_version,
        "ffmpeg_build_config": ffmpeg_build_config,
        "ffmpeg_returncode": ffmpeg_rc,
        "ffmpeg_encoders_returncode": encoders_rc,
        "nvidia_smi_returncode": gpu_rc,
        "gpu_info": gpu_rows,
        "primary_gpu": {
            "gpu_name": first_gpu.get("name", ""),
            "gpu_driver_version": first_gpu.get("driver_version", ""),
            "gpu_memory_total_mb": first_gpu.get("memory_total_mb", ""),
            "gpu_index": first_gpu.get("index", 0) if first_gpu else "",
        },
        "encoder_checks": encoder_checks,
        "sources": sources,
    }
    (env_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return environment


def dry_run_environment(source_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "host_name": socket.gethostname(),
        "os_name": platform.system(),
        "os_version": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "missing_tools": [],
        "ffmpeg_path": FFMPEG_PATH,
        "ffprobe_path": FFPROBE_PATH,
        "ffmpeg_version": "",
        "ffmpeg_build_config": "",
        "primary_gpu": {
            "gpu_name": "",
            "gpu_driver_version": "",
            "gpu_memory_total_mb": "",
            "gpu_index": "",
        },
        "encoder_checks": {
            codec_label: {
                "encoder": encoder,
                "available": True,
                "preset_slow_supported": True,
            }
            for codec_label, encoder in CODEC_ENCODERS.items()
        },
        "sources": {
            label: {
                "path": str(path),
                "filename": path.name,
                "width": "",
                "height": "",
                "fps": "",
                "duration_sec": 120 if label == "fhd" else 30,
                "video_codec": "",
                "video_bitrate_mbps": "",
                "pix_fmt": "",
                "file_size_mb": "",
            }
            for label, path in source_paths.items()
        },
    }
