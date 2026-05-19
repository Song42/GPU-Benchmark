#!/usr/bin/env python3
"""Main entrypoint for the GPU video encode benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_config import (
    AXIS_BITRATE,
    AXIS_CQ23,
    BITRATE_MATRIX,
    CODEC_ENCODERS,
    CODEC_LABELS,
    CQ_VALUE,
    PARALLEL_JOBS_DEFAULT,
    PRESET,
    STATUS_FAIL_PRESET_UNSUPPORTED,
    STATUS_FAIL_UNSUPPORTED_ENCODER,
    STATUS_FAIL_VMAF,
    STATUS_SKIPPED_PRECHECK_FAILED,
    STATUS_SUCCESS,
    bitrate_bufsize_mbps,
    parallel_timeout_sec,
    single_timeout_sec,
)
from commands import (
    build_encode_command,
    build_vmaf_command,
    command_to_string,
    failed_log_filename,
    output_filename,
    slugify,
)
from monitor import monitor_aggregates, parse_monitor_rows, start_monitor, stop_monitor
from parser import parse_vmaf_json
from preflight import collect_environment, dry_run_environment
from report import RUNS_COLUMNS, VMAF_COLUMNS, blank_row, write_all_reports
from runner import JobSpec, run_parallel_jobs, run_vmaf_command, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NVIDIA GPU encoding benchmarks and emit CSV reports."
    )
    parser.add_argument("--source-uhd", required=True, type=Path)
    parser.add_argument("--source-fhd", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-label", required=True)
    parser.add_argument(
        "--parallel-levels",
        default=",".join(str(value) for value in PARALLEL_JOBS_DEFAULT),
        help="Comma-separated parallel levels.",
    )
    parser.add_argument("--skip-vmaf", action="store_true")
    parser.add_argument(
        "--only-axis",
        choices=[AXIS_BITRATE, AXIS_CQ23, "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_parallel_levels(value: str) -> list[int]:
    levels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        level = int(part)
        if level < 1:
            raise ValueError("parallel levels must be >= 1")
        levels.append(level)
    return levels or PARALLEL_JOBS_DEFAULT


def make_dirs(output_dir: Path, gpu_slug: str) -> dict[str, Path]:
    paths = {
        "env": output_dir / "env",
        "outputs_bitrate": output_dir / "outputs" / gpu_slug / AXIS_BITRATE,
        "outputs_cq23": output_dir / "outputs" / gpu_slug / AXIS_CQ23,
        "logs_ffmpeg": output_dir / "logs" / "ffmpeg",
        "logs_gpu_monitor": output_dir / "logs" / "gpu_monitor",
        "logs_vmaf": output_dir / "logs" / "vmaf",
        "csv": output_dir / "csv",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_cases(args: argparse.Namespace, environment: dict[str, Any]) -> list[dict[str, Any]]:
    source_paths = {"uhd": args.source_uhd, "fhd": args.source_fhd}
    axes = [AXIS_BITRATE, AXIS_CQ23] if args.only_axis == "all" else [args.only_axis]
    cases: list[dict[str, Any]] = []
    for source_label, source_path in source_paths.items():
        metadata = environment["sources"][source_label]
        target_resolution = (
            f"{metadata.get('width')}x{metadata.get('height')}"
            if metadata.get("width") and metadata.get("height")
            else source_label.upper()
        )
        if AXIS_BITRATE in axes:
            for bitrate in BITRATE_MATRIX[source_label]:
                for codec_label in CODEC_LABELS:
                    cases.append(
                        {
                            "benchmark_axis": AXIS_BITRATE,
                            "source_label": source_label,
                            "source_path": source_path,
                            "source_metadata": metadata,
                            "target_resolution": target_resolution,
                            "codec_label": codec_label,
                            "encoder": CODEC_ENCODERS[codec_label],
                            "preset": PRESET,
                            "rate_control": "cbr",
                            "target_bitrate_mbps": bitrate,
                            "maxrate_mbps": bitrate,
                            "bufsize_mbps": bitrate_bufsize_mbps(bitrate),
                            "cq": "",
                        }
                    )
        if AXIS_CQ23 in axes:
            for codec_label in CODEC_LABELS:
                cases.append(
                    {
                        "benchmark_axis": AXIS_CQ23,
                        "source_label": source_label,
                        "source_path": source_path,
                        "source_metadata": metadata,
                        "target_resolution": target_resolution,
                        "codec_label": codec_label,
                        "encoder": CODEC_ENCODERS[codec_label],
                        "preset": PRESET,
                        "rate_control": "vbr",
                        "target_bitrate_mbps": "",
                        "maxrate_mbps": "",
                        "bufsize_mbps": "",
                        "cq": CQ_VALUE,
                    }
                )
    return cases


def base_run_row(
    environment: dict[str, Any],
    args: argparse.Namespace,
    case: dict[str, Any],
    batch_id: str,
    job_id: str,
    parallel_jobs: int,
    job_index: int,
    command: list[str],
    output_path: Path,
) -> dict[str, Any]:
    row = blank_row(RUNS_COLUMNS)
    gpu = environment.get("primary_gpu", {})
    source = case["source_metadata"]
    row.update(
        {
            "run_id": job_id,
            "batch_id": batch_id,
            "job_id": job_id,
            "host_name": environment.get("host_name", ""),
            "os_name": environment.get("os_name", ""),
            "os_version": environment.get("os_version", ""),
            "python_version": environment.get("python_version", ""),
            "gpu_label": args.gpu_label,
            "gpu_name": gpu.get("gpu_name", ""),
            "gpu_index": gpu.get("gpu_index", ""),
            "gpu_driver_version": gpu.get("gpu_driver_version", ""),
            "gpu_memory_total_mb": gpu.get("gpu_memory_total_mb", ""),
            "ffmpeg_version": environment.get("ffmpeg_version", ""),
            "ffmpeg_build_config": environment.get("ffmpeg_build_config", ""),
            "input_path": str(case["source_path"]),
            "input_filename": Path(case["source_path"]).name,
            "input_resolution_width": source.get("width", ""),
            "input_resolution_height": source.get("height", ""),
            "input_fps": source.get("fps", ""),
            "input_duration_sec": source.get("duration_sec", ""),
            "input_video_codec": source.get("video_codec", ""),
            "input_video_bitrate_mbps": source.get("video_bitrate_mbps", ""),
            "input_pix_fmt": source.get("pix_fmt", ""),
            "input_file_size_mb": source.get("file_size_mb", ""),
            "benchmark_axis": case["benchmark_axis"],
            "source_label": case["source_label"],
            "target_resolution": case["target_resolution"],
            "codec_label": case["codec_label"],
            "encoder": case["encoder"],
            "preset": case["preset"],
            "rate_control": case["rate_control"],
            "target_bitrate_mbps": case["target_bitrate_mbps"],
            "maxrate_mbps": case["maxrate_mbps"],
            "bufsize_mbps": case["bufsize_mbps"],
            "cq": case["cq"],
            "parallel_jobs": parallel_jobs,
            "parallel_job_index": job_index,
            "command": command_to_string(command),
            "output_path": str(output_path),
            "output_filename": output_path.name,
            "output_exists": False,
        }
    )
    return row


def synthetic_failure_row(
    row: dict[str, Any],
    status: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    now = utc_now()
    row.update(
        {
            "timestamp_start": now,
            "timestamp_end": now,
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "wall_time_sec": 0,
            "batch_time_sec": 0,
            "output_exists": False,
        }
    )
    return row


def enrich_row_with_result(
    row: dict[str, Any],
    result: dict[str, Any],
    monitor_stats: dict[str, Any],
) -> dict[str, Any]:
    output = result.pop("output_metadata", {}) or {}
    row.update(result)
    row.update(
        {
            "output_duration_sec": output.get("duration_sec", ""),
            "output_width": output.get("width", ""),
            "output_height": output.get("height", ""),
            "output_fps": output.get("fps", ""),
            "output_video_codec": output.get("video_codec", ""),
            "output_pix_fmt": output.get("pix_fmt", ""),
            "output_video_bitrate_mbps": output.get("video_bitrate_mbps", ""),
        }
    )
    row.update(monitor_stats)
    return row


def case_output_dir(paths: dict[str, Path], case: dict[str, Any]) -> Path:
    return paths["outputs_bitrate"] if case["benchmark_axis"] == AXIS_BITRATE else paths["outputs_cq23"]


def run_encoding_phase(
    args: argparse.Namespace,
    environment: dict[str, Any],
    paths: dict[str, Path],
    cases: list[dict[str, Any]],
    parallel_levels: list[int],
    gpu_slug: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    gpu_monitor_rows: list[dict[str, Any]] = []
    encoder_checks = environment.get("encoder_checks", {})

    for case in cases:
        duration = case["source_metadata"].get("duration_sec") or 0
        for parallel_jobs in parallel_levels:
            timeout = (
                single_timeout_sec(float(duration) if duration else None)
                if parallel_jobs == 1
                else parallel_timeout_sec(float(duration) if duration else None)
            )
            batch_id = uuid.uuid4().hex
            job_specs: list[JobSpec] = []
            base_rows: list[dict[str, Any]] = []
            for job_index in range(parallel_jobs):
                job_id = uuid.uuid4().hex
                output_path = case_output_dir(paths, case) / output_filename(
                    gpu_slug, case, parallel_jobs, job_index
                )
                command = build_encode_command(case, case["source_path"], output_path)
                row = base_run_row(
                    environment,
                    args,
                    case,
                    batch_id,
                    job_id,
                    parallel_jobs,
                    job_index,
                    command,
                    output_path,
                )
                base_rows.append(row)
                job_specs.append(
                    JobSpec(
                        job_id=job_id,
                        job_index=job_index,
                        command=command,
                        output_path=output_path,
                        log_path=paths["logs_ffmpeg"]
                        / output_filename(gpu_slug, case, parallel_jobs, job_index).replace(
                            ".mp4", ".log"
                        ),
                    )
                )

            if (
                not args.overwrite
                and not args.dry_run
                and any(Path(str(row["output_path"])).exists() for row in base_rows)
            ):
                runs.extend(
                    synthetic_failure_row(
                        row,
                        STATUS_SKIPPED_PRECHECK_FAILED,
                        "output_exists",
                        "Output already exists; rerun with --overwrite to replace it",
                    )
                    for row in base_rows
                )
                continue

            check = encoder_checks.get(case["codec_label"], {})
            if not check.get("available", False):
                runs.extend(
                    synthetic_failure_row(
                        row,
                        STATUS_FAIL_UNSUPPORTED_ENCODER,
                        "unsupported_encoder",
                        f"{case['encoder']} is not listed by ffmpeg -encoders",
                    )
                    for row in base_rows
                )
                failed_log = paths["logs_ffmpeg"] / failed_log_filename(
                    gpu_slug, case, parallel_jobs, 0
                )
                failed_log.write_text(
                    f"{case['encoder']} is not listed by ffmpeg -encoders\n",
                    encoding="utf-8",
                )
                continue
            if not check.get("preset_slow_supported", False):
                runs.extend(
                    synthetic_failure_row(
                        row,
                        STATUS_FAIL_PRESET_UNSUPPORTED,
                        "preset_unsupported",
                        f"preset {PRESET} is not listed for {case['encoder']}",
                    )
                    for row in base_rows
                )
                continue
            if args.dry_run:
                runs.extend(
                    synthetic_failure_row(
                        row,
                        STATUS_SKIPPED_PRECHECK_FAILED,
                        "dry_run",
                        "Dry run: command was not executed",
                    )
                    for row in base_rows
                )
                continue

            monitor_handle = start_monitor(
                paths["logs_gpu_monitor"], batch_id, parallel_jobs
            )
            results, _ = run_parallel_jobs(job_specs, timeout, case["encoder"])
            stop_monitor(monitor_handle)
            monitor_rows = parse_monitor_rows(
                monitor_handle, base_rows[0]["run_id"], batch_id, parallel_jobs
            )
            gpu_monitor_rows.extend(monitor_rows)
            monitor_stats = monitor_aggregates(monitor_rows)
            result_by_job_id = {result["job_id"]: result for result in results}
            for row in base_rows:
                result = result_by_job_id.get(row["job_id"], {})
                runs.append(enrich_row_with_result(row, result, monitor_stats))
    return runs, gpu_monitor_rows


def run_vmaf_phase(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    vmaf_rows: list[dict[str, Any]] = []
    if args.skip_vmaf or args.dry_run:
        return vmaf_rows

    for row in runs:
        if row.get("status") != STATUS_SUCCESS:
            continue
        output_path = Path(str(row["output_path"]))
        source_path = Path(str(row["input_path"]))
        vmaf_json_path = (
            paths["logs_vmaf"] / f"{output_path.stem}_vmaf.json"
        )
        vmaf_log_path = paths["logs_vmaf"] / f"{output_path.stem}_vmaf.log"
        row["vmaf_json_path"] = str(vmaf_json_path)
        command = build_vmaf_command(output_path, source_path, vmaf_json_path)
        timeout = parallel_timeout_sec(float(row.get("input_duration_sec") or 0))
        ok, _, error = run_vmaf_command(command, timeout, vmaf_log_path)
        vmaf_row = blank_row(VMAF_COLUMNS)
        vmaf_row.update(
            {
                "run_id": row["run_id"],
                "encoded_path": str(output_path),
                "source_path": str(source_path),
                "vmaf_json_path": str(vmaf_json_path),
                "vmaf_status": STATUS_SUCCESS if ok else STATUS_FAIL_VMAF,
                "vmaf_error_message": error,
            }
        )
        if ok:
            try:
                metrics = parse_vmaf_json(vmaf_json_path)
                vmaf_row.update(metrics)
                row.update(
                    {
                        "vmaf_mean": metrics.get("vmaf_mean", ""),
                        "vmaf_min": metrics.get("vmaf_min", ""),
                        "vmaf_harmonic_mean": metrics.get("vmaf_harmonic_mean", ""),
                    }
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                vmaf_row["vmaf_status"] = STATUS_FAIL_VMAF
                vmaf_row["vmaf_error_message"] = str(exc)
        vmaf_rows.append(vmaf_row)
    return vmaf_rows


def dry_run_plan(cases: list[dict[str, Any]], parallel_levels: list[int]) -> None:
    total_jobs = len(cases) * sum(parallel_levels)
    print(f"Dry run: {len(cases)} benchmark cases, {total_jobs} encode jobs.")
    for case in cases[:6]:
        print(
            f"  {case['benchmark_axis']} {case['source_label']} "
            f"{case['codec_label']} {case['rate_control']}"
        )
    if len(cases) > 6:
        print(f"  ... {len(cases) - 6} more cases")


def main() -> int:
    args = parse_args()
    parallel_levels = parse_parallel_levels(args.parallel_levels)
    gpu_slug = slugify(args.gpu_label)
    paths = make_dirs(args.output_dir, gpu_slug)
    source_paths = {"uhd": args.source_uhd, "fhd": args.source_fhd}

    if args.dry_run:
        environment = dry_run_environment(source_paths)
    else:
        environment = collect_environment(paths["env"], source_paths)

    cases = build_cases(args, environment)
    if args.dry_run:
        dry_run_plan(cases, parallel_levels)

    runs, gpu_monitor_rows = run_encoding_phase(
        args, environment, paths, cases, parallel_levels, gpu_slug
    )
    vmaf_rows = run_vmaf_phase(args, runs, paths)
    write_all_reports(paths["csv"], runs, vmaf_rows, gpu_monitor_rows)

    if args.dry_run:
        print(f"Dry-run CSVs written under {paths['csv']}")
    else:
        print(f"Benchmark complete. CSVs written under {paths['csv']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
