"""CSV report writers and summary calculations."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench_config import STATUS_SUCCESS, scaling_verdict
from parser import average, maximum, safe_float


RUNS_COLUMNS = [
    "run_id",
    "batch_id",
    "job_id",
    "timestamp_start",
    "timestamp_end",
    "host_name",
    "os_name",
    "os_version",
    "python_version",
    "gpu_label",
    "gpu_name",
    "gpu_index",
    "gpu_driver_version",
    "gpu_memory_total_mb",
    "ffmpeg_version",
    "ffmpeg_build_config",
    "input_path",
    "input_filename",
    "input_resolution_width",
    "input_resolution_height",
    "input_fps",
    "input_duration_sec",
    "input_video_codec",
    "input_video_bitrate_mbps",
    "input_pix_fmt",
    "input_file_size_mb",
    "benchmark_axis",
    "source_label",
    "target_resolution",
    "codec_label",
    "encoder",
    "preset",
    "rate_control",
    "target_bitrate_mbps",
    "maxrate_mbps",
    "bufsize_mbps",
    "cq",
    "parallel_jobs",
    "parallel_job_index",
    "command",
    "output_path",
    "output_filename",
    "status",
    "exit_code",
    "error_type",
    "error_message",
    "ffmpeg_rtime_sec",
    "ffmpeg_utime_sec",
    "ffmpeg_stime_sec",
    "wall_time_sec",
    "batch_time_sec",
    "output_exists",
    "output_file_size_mb",
    "output_duration_sec",
    "output_width",
    "output_height",
    "output_fps",
    "output_video_codec",
    "output_pix_fmt",
    "output_video_bitrate_mbps",
    "speed_x",
    "frames_encoded",
    "avg_fps",
    "vmaf_json_path",
    "vmaf_mean",
    "vmaf_min",
    "vmaf_harmonic_mean",
    "gpu_util_avg_percent",
    "gpu_util_max_percent",
    "encoder_util_avg_percent",
    "encoder_util_max_percent",
    "decoder_util_avg_percent",
    "memory_used_avg_mb",
    "memory_used_max_mb",
    "power_draw_avg_w",
    "temperature_avg_c",
    "notes",
]

VMAF_COLUMNS = [
    "run_id",
    "encoded_path",
    "source_path",
    "vmaf_json_path",
    "vmaf_mean",
    "vmaf_min",
    "vmaf_max",
    "vmaf_harmonic_mean",
    "vmaf_p1",
    "vmaf_p5",
    "vmaf_p95",
    "vmaf_p99",
    "vmaf_status",
    "vmaf_error_message",
]

GPU_MONITOR_COLUMNS = [
    "timestamp",
    "gpu_name",
    "gpu_index",
    "util_gpu_percent",
    "util_memory_percent",
    "util_encoder_percent",
    "util_decoder_percent",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "temperature_c",
    "clocks_sm_mhz",
    "clocks_mem_mhz",
    "run_id",
    "batch_id",
    "parallel_jobs",
]

SUMMARY_COLUMNS = [
    "gpu_label",
    "gpu_name",
    "benchmark_axis",
    "source_label",
    "resolution",
    "codec_label",
    "encoder",
    "preset",
    "rate_control",
    "target_bitrate_mbps",
    "cq",
    "parallel_jobs",
    "total_jobs",
    "completed_jobs",
    "failed_jobs",
    "batch_time_sec",
    "avg_individual_rtime_sec",
    "min_individual_rtime_sec",
    "max_individual_rtime_sec",
    "throughput_jobs_per_sec",
    "throughput_fps",
    "speedup_vs_single",
    "scaling_efficiency",
    "avg_output_bitrate_mbps",
    "avg_output_size_mb",
    "avg_vmaf_mean",
    "avg_gpu_util_percent",
    "avg_encoder_util_percent",
    "avg_memory_used_mb",
    "status_summary",
    "scaling_verdict",
]


def blank_row(columns: list[str]) -> dict[str, Any]:
    return {column: "" for column in columns}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in columns})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return value


def generate_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = [
        "gpu_label",
        "gpu_name",
        "benchmark_axis",
        "source_label",
        "target_resolution",
        "codec_label",
        "encoder",
        "preset",
        "rate_control",
        "target_bitrate_mbps",
        "cq",
        "parallel_jobs",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    base_throughputs: dict[tuple[Any, ...], float | None] = {}
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        key_map = dict(zip(group_keys, key))
        completed = [row for row in rows if row.get("status") == STATUS_SUCCESS]
        batch_time = maximum([row.get("batch_time_sec") for row in rows])
        total_jobs = len(rows)
        completed_jobs = len(completed)
        throughput = (
            completed_jobs / batch_time if batch_time and batch_time > 0 else None
        )
        throughput_fps = average(
            [
                (row.get("frames_encoded") or 0) / row.get("wall_time_sec")
                for row in completed
                if row.get("wall_time_sec")
            ]
        )
        base_key = tuple(
            key_map.get(name)
            for name in [
                "gpu_label",
                "gpu_name",
                "benchmark_axis",
                "source_label",
                "target_resolution",
                "codec_label",
                "encoder",
                "preset",
                "rate_control",
                "target_bitrate_mbps",
                "cq",
            ]
        )
        if int(key_map["parallel_jobs"]) == 1:
            base_throughputs[base_key] = throughput
        base_throughput = base_throughputs.get(base_key)
        speedup = throughput / base_throughput if throughput and base_throughput else None
        efficiency = (
            speedup / int(key_map["parallel_jobs"])
            if speedup is not None and int(key_map["parallel_jobs"])
            else None
        )
        status_counts = Counter(str(row.get("status") or "") for row in rows)
        status_summary = ";".join(
            f"{status}:{count}" for status, count in sorted(status_counts.items())
        )
        summaries.append(
            {
                "gpu_label": key_map["gpu_label"],
                "gpu_name": key_map["gpu_name"],
                "benchmark_axis": key_map["benchmark_axis"],
                "source_label": key_map["source_label"],
                "resolution": key_map["target_resolution"],
                "codec_label": key_map["codec_label"],
                "encoder": key_map["encoder"],
                "preset": key_map["preset"],
                "rate_control": key_map["rate_control"],
                "target_bitrate_mbps": key_map["target_bitrate_mbps"],
                "cq": key_map["cq"],
                "parallel_jobs": key_map["parallel_jobs"],
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": total_jobs - completed_jobs,
                "batch_time_sec": batch_time,
                "avg_individual_rtime_sec": average(
                    [row.get("ffmpeg_rtime_sec") for row in completed]
                ),
                "min_individual_rtime_sec": min(
                    [
                        safe_float(row.get("ffmpeg_rtime_sec"))
                        for row in completed
                        if safe_float(row.get("ffmpeg_rtime_sec")) is not None
                    ],
                    default=None,
                ),
                "max_individual_rtime_sec": maximum(
                    [row.get("ffmpeg_rtime_sec") for row in completed]
                ),
                "throughput_jobs_per_sec": throughput,
                "throughput_fps": throughput_fps,
                "speedup_vs_single": speedup,
                "scaling_efficiency": efficiency,
                "avg_output_bitrate_mbps": average(
                    [row.get("output_video_bitrate_mbps") for row in completed]
                ),
                "avg_output_size_mb": average(
                    [row.get("output_file_size_mb") for row in completed]
                ),
                "avg_vmaf_mean": average([row.get("vmaf_mean") for row in completed]),
                "avg_gpu_util_percent": average(
                    [row.get("gpu_util_avg_percent") for row in rows]
                ),
                "avg_encoder_util_percent": average(
                    [row.get("encoder_util_avg_percent") for row in rows]
                ),
                "avg_memory_used_mb": average(
                    [row.get("memory_used_avg_mb") for row in rows]
                ),
                "status_summary": status_summary,
                "scaling_verdict": scaling_verdict(
                    efficiency, dict(status_counts), total_jobs
                ),
            }
        )
    return summaries


def write_all_reports(
    csv_dir: Path,
    runs: list[dict[str, Any]],
    vmaf_rows: list[dict[str, Any]],
    gpu_monitor_rows: list[dict[str, Any]],
) -> None:
    write_csv(csv_dir / "runs.csv", runs, RUNS_COLUMNS)
    write_csv(csv_dir / "vmaf.csv", vmaf_rows, VMAF_COLUMNS)
    write_csv(csv_dir / "gpu_monitor.csv", gpu_monitor_rows, GPU_MONITOR_COLUMNS)
    write_csv(csv_dir / "summary.csv", generate_summary(runs), SUMMARY_COLUMNS)
